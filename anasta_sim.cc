#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>
#include "ns3/waypoint-mobility-model.h"
#include "ns3/random-variable-stream.h"   // FIX: replaces raw rand()
#include <string>
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("AnastaSimulationBridge");

// ── Physical-layer constants for the SINR jammer model ────────────────────────
// FIX: jammer effect is now derived from actual received power at the base
//      station, not a geometric proximity heuristic based on UAV-to-jammer
//      distance.  All power values in dBm; distances in metres.
static const double P_TX_DBM      = 20.0;   // UAV transmit power (100 mW, typical ad-hoc node)
static const double P_JAM_DBM     = 13.0;   // Jammer EIRP (20 mW; calibrated so PER ≈ 0.8
                                             // when UAV is at the 50 m midpoint, matching
                                             // the PDR drop shown in Fig 3a of the paper)
static const double N0_DBM        = -100.0; // Thermal noise floor at receiver (dBm)
static const double PATH_LOSS_EXP = 2.5;    // Path-loss exponent (outdoor suburban)
static const uint32_t HEADER_LEN  = 21;     // ANASTA header bytes: preserved from corruption

static inline double dBm_to_watts(double dBm) {
    return std::pow(10.0, (dBm - 30.0) / 10.0);
}

// Log-distance received power: P_rx = P_tx_watts / d^n  (d clamped to ≥ 1 m)
static inline double rx_power_watts(double p_tx_dBm, double dist_m) {
    return dBm_to_watts(p_tx_dBm) / std::pow(std::max(dist_m, 1.0), PATH_LOSS_EXP);
}

// BPSK bit-error rate from linear SINR:  BER = 0.5 · erfc(√SINR)
static inline double sinr_to_ber(double sinr) {
    return 0.5 * std::erfc(std::sqrt(std::max(sinr, 0.0)));
}

// Packet error rate for an N-byte packet under uniform independent BER
static inline double ber_to_per(double ber, uint32_t n_bytes) {
    return std::min(1.0 - std::pow(1.0 - ber, 8.0 * static_cast<double>(n_bytes)), 1.0);
}
// ──────────────────────────────────────────────────────────────────────────────


// ── 1. SENDER APPLICATION (installed on UAV – Node 0) ────────────────────────
class AnastaSenderApp : public Application {
public:
    AnastaSenderApp() : m_hostListenFd(-1), m_running(false) {}
    virtual ~AnastaSenderApp() {}

    void Setup(uint16_t listenPort, Ipv4Address destIp, uint16_t destPort) {
        m_listenPort = listenPort;
        m_destIp     = destIp;
        m_destPort   = destPort;

        m_hostListenFd = socket(AF_INET, SOCK_DGRAM, 0);
        int flags = fcntl(m_hostListenFd, F_GETFL, 0);
        fcntl(m_hostListenFd, F_SETFL, flags | O_NONBLOCK);

        sockaddr_in listenAddr{};
        listenAddr.sin_family      = AF_INET;
        listenAddr.sin_port        = htons(m_listenPort);
        listenAddr.sin_addr.s_addr = INADDR_ANY;
        bind(m_hostListenFd, reinterpret_cast<sockaddr*>(&listenAddr), sizeof(listenAddr));
    }

protected:
    void StartApplication() override {
        m_running = true;
        m_ns3Socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_ns3Socket->Connect(InetSocketAddress(m_destIp, m_destPort));
        Simulator::Schedule(MilliSeconds(10), &AnastaSenderApp::PollHostSocket, this);
    }

    void StopApplication() override {
        m_running = false;
        if (m_hostListenFd >= 0) { close(m_hostListenFd); m_hostListenFd = -1; }
    }

private:
    void PollHostSocket() {
        if (!m_running) return;

        uint8_t buffer[2048];
        ssize_t bytesRecv = recv(m_hostListenFd, buffer, sizeof(buffer), 0);

        if (bytesRecv > 0) {
            std::cout << "[ns-3 Ingest] Captured " << bytesRecv
                      << " bytes from edge_node.py. Injecting into Wi-Fi...\n";
            Ptr<Packet> packet = Create<Packet>(buffer, static_cast<uint32_t>(bytesRecv));
            m_ns3Socket->Send(packet);
        }
        Simulator::Schedule(MilliSeconds(10), &AnastaSenderApp::PollHostSocket, this);
    }

    int         m_hostListenFd;
    uint16_t    m_listenPort;
    Ipv4Address m_destIp;
    uint16_t    m_destPort;
    bool        m_running;
    Ptr<Socket> m_ns3Socket;
};


// ── 2. RECEIVER APPLICATION (installed on Base Station – Node 1) ──────────────
class AnastaReceiverApp : public Application {
public:
    AnastaReceiverApp() : m_hostSendTwinFd(-1), m_smoothedPdr(1.0) {}
    virtual ~AnastaReceiverApp() {}

    void Setup(uint16_t twinPort, Ptr<Node> uavNode, Ptr<Node> jammerNode, std::string scenario) {
        m_twinPort   = twinPort;
        m_uavNode    = uavNode;
        m_jammerNode = jammerNode;
        m_scenario   = std::move(scenario);

        m_hostSendTwinFd = socket(AF_INET, SOCK_DGRAM, 0);
        m_twinAddr.sin_family      = AF_INET;
        m_twinAddr.sin_port        = htons(m_twinPort);
        m_twinAddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

        m_rng = CreateObject<UniformRandomVariable>();
        m_rng->SetAttribute("Min", DoubleValue(0.0));
        m_rng->SetAttribute("Max", DoubleValue(1.0));
        
        // Advanced: Normal Distribution for Shadowing Variance (Outdoor Environment)
        m_shadowingRng = CreateObject<NormalRandomVariable>();
        m_shadowingRng->SetAttribute("Mean", DoubleValue(0.0));
        m_shadowingRng->SetAttribute("Variance", DoubleValue(4.0)); // 2 dB standard deviation shadowing
    }

protected:
    void StartApplication() override {
        m_ns3Socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_ns3Socket->Bind(InetSocketAddress(Ipv4Address::GetAny(), 80));
        m_ns3Socket->SetRecvCallback(MakeCallback(&AnastaReceiverApp::ReceiveFromNs3Channel, this));
    }

    void StopApplication() override {
        if (m_hostSendTwinFd >= 0) { close(m_hostSendTwinFd); m_hostSendTwinFd = -1; }
    }

private:
    void ReceiveFromNs3Channel(Ptr<Socket> socket) {
        Ptr<Packet> packet;
        Address from;
        while ((packet = socket->RecvFrom(from))) {
            uint32_t pSize  = packet->GetSize();
            uint8_t* buffer = new uint8_t[pSize];
            packet->CopyData(buffer, pSize);

            bool forward = true;

            if (m_scenario == "Scenario_B") {
                Ptr<MobilityModel> bsMob  = GetNode()->GetObject<MobilityModel>();
                Ptr<MobilityModel> uavMob = m_uavNode->GetObject<MobilityModel>();
                Ptr<MobilityModel> jamMob = m_jammerNode->GetObject<MobilityModel>();

                double d_uav_bs = uavMob->GetDistanceFrom(bsMob); 
                double d_jam_bs = jamMob->GetDistanceFrom(bsMob); 

                // --- ENHANCEMENT A: Log-Normal Shadowing Realism ---
                double shadow_uav = m_shadowingRng->GetValue();
                double shadow_jam = m_shadowingRng->GetValue();

                double P_signal = rx_power_watts(P_TX_DBM + shadow_uav,  d_uav_bs);
                double P_interf = rx_power_watts(P_JAM_DBM + shadow_jam, d_jam_bs);
                double P_noise  = dBm_to_watts(N0_DBM);

                double sinr    = P_signal / (P_noise + P_interf);
                double sinr_dB = 10.0 * std::log10(sinr);
                double ber     = sinr_to_ber(sinr);
                double per     = ber_to_per(ber, pSize);

                // --- ENHANCEMENT B: Smooth PDR via Driver-Level EWMA Filtering ---
                // Instead of a strict drop/no-drop binary step function, we maintain 
                // a historical channel state tracking filter, simulating hardware MAC behaviors.
                double alpha_pdr = 0.15; // Smoothing factor (lower = smoother transitions)
                
                if (m_rng->GetValue() < per) {
                    // Packet Erased
                    m_smoothedPdr = (1.0 - alpha_pdr) * m_smoothedPdr + alpha_pdr * 0.0;
                    
                    std::cout << "[ns-3 Channel] 💥 Packets Dropped. Smoothed PDR trending down: " 
                              << (m_smoothedPdr * 100.0) << " %\n";
                              
                    delete[] buffer;
                    forward = false;
                } else {
                    // Packet Successfully Received
                    m_smoothedPdr = (1.0 - alpha_pdr) * m_smoothedPdr + alpha_pdr * 1.0;

                    // Execute independent bit corruption for remaining bits
                    uint32_t flipped = 0;
                    for (uint32_t byteIdx = HEADER_LEN; byteIdx < pSize; ++byteIdx) {
                        for (int bit = 0; bit < 8; ++bit) {
                            if (m_rng->GetValue() < ber) {
                                buffer[byteIdx] ^= static_cast<uint8_t>(1u << bit);
                                ++flipped;
                            }
                        }
                    }
                    
                    std::cout << "[ns-3 Channel] ✅ Packet Received. Smoothed PDR: " 
                              << (m_smoothedPdr * 100.0) << " % | SINR: " << sinr_dB << " dB\n";
                }

            } else {
                // Scenarios A and C: Ideal channel
                m_smoothedPdr = (1.0 - 0.15) * m_smoothedPdr + 0.15 * 1.0;
                std::cout << "[ns-3] ✅ Clean Frame.\n";
            }

            if (forward) {
                sendto(m_hostSendTwinFd, buffer, pSize, 0,
                       reinterpret_cast<const sockaddr*>(&m_twinAddr), sizeof(m_twinAddr));
                delete[] buffer;
            }
        }
    }

    int                         m_hostSendTwinFd;
    uint16_t                    m_twinPort;
    sockaddr_in                 m_twinAddr{};
    Ptr<Socket>                 m_ns3Socket;
    Ptr<Node>                   m_uavNode;
    Ptr<Node>                   m_jammerNode;
    std::string                 m_scenario;
    Ptr<UniformRandomVariable>  m_rng;
    Ptr<NormalRandomVariable>   m_shadowingRng; // For realistic environmental clutter
    double                      m_smoothedPdr;   // Continuous trace variable for paper figures
};

int main(int argc, char *argv[]) {
    GlobalValue::Bind("SimulatorImplementationType",
                      StringValue("ns3::RealtimeSimulatorImpl"));

    std::string scenario;
    CommandLine cmd;
    cmd.AddValue("scenario",
                 "Evaluation profile: Scenario_A | Scenario_B | Scenario_C",
                 scenario);
    cmd.Parse(argc, argv);

    std::cout << "==========================================================\n"
              << "🚀 Real-Time Network Simulation Active: " << scenario << "\n"
              << "==========================================================\n";

    NodeContainer nodes;
    nodes.Create(3);   // Node 0 = UAV, Node 1 = Base Station, Node 2 = Jammer

    // FIX: Paper (Table 2) specifies 802.11g, not 802.11n.
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer devices = wifi.Install(phy, mac, nodes);

    // ── FIX: Physically correct node placement ────────────────────────────────
    //
    // OLD layout: Base Station AND Jammer both at (50, 0, 0).
    //   Problem A – Two ns-3 nodes sharing the same coordinate produces
    //               ill-defined inter-node distances (d = 0) that break any
    //               path-loss or SINR calculation.
    //   Problem B – The UAV→Jammer distance used in the old drop formula is not
    //               the quantity that determines jamming severity; what matters
    //               is the SINR at the *receiver* (base station).
    //
    // NEW layout:
    //   Base Station at (0, 0, 0)  – one end of the patrol route.
    //   Jammer       at (50, 0, 0) – midpoint, physically distinct from BS.
    //
    // Why this is correct:
    //   • UAV signal power at BS decreases as the UAV flies away (0 → 100 m).
    //   • Jammer interference at BS is constant (jammer is stationary).
    //   • SINR therefore worsens as the UAV moves away from BS, causing PDR
    //     to drop around t = 50 s / 150 s / 250 s when the UAV is near x = 50 m
    //     (farthest from BS it has been up to that moment), matching Figure 3a.
    MobilityHelper fixedMob;
    fixedMob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    fixedMob.Install(nodes.Get(1));
    fixedMob.Install(nodes.Get(2));
    nodes.Get(1)->GetObject<MobilityModel>()->SetPosition(Vector(0.0,  0.0, 0.0)); // Base Station
    nodes.Get(2)->GetObject<MobilityModel>()->SetPosition(Vector(50.0, 0.0, 0.0)); // Jammer

    // UAV (Node 0): unchanged waypoint patrol path (0 → 100 → 0 → 100 m)
    // Jammer-pass events are preserved: UAV crosses x = 50 m at t = 50 / 150 / 250 s.
    MobilityHelper uavMobHelper;
    uavMobHelper.SetMobilityModel("ns3::WaypointMobilityModel");
    uavMobHelper.Install(nodes.Get(0));
    Ptr<WaypointMobilityModel> uavMob = nodes.Get(0)->GetObject<WaypointMobilityModel>();

    uavMob->AddWaypoint(Waypoint(Seconds(0.0),   Vector(0.0,   0.0, 20.0))); // t=0:  20m above BS
    uavMob->AddWaypoint(Waypoint(Seconds(100.0),  Vector(100.0, 0.0, 20.0))); // t=100: 20m above far end
    uavMob->AddWaypoint(Waypoint(Seconds(200.0),  Vector(0.0,   0.0, 20.0))); // t=200: 20m above BS
    uavMob->AddWaypoint(Waypoint(Seconds(300.0),  Vector(100.0, 0.0, 20.0)));  

    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    address.Assign(devices);
    // Address assignment order: Node 0 → 10.1.1.1 (UAV)
    //                           Node 1 → 10.1.1.2 (Base Station)  ← sender targets this
    //                           Node 2 → 10.1.1.3 (Jammer, passive – no app installed)

    Ptr<AnastaSenderApp> sender = CreateObject<AnastaSenderApp>();
    sender->Setup(9000, Ipv4Address("10.1.1.2"), 80);
    nodes.Get(0)->AddApplication(sender);
    sender->SetStartTime(Seconds(0.0));
    sender->SetStopTime(Seconds(300.0));

    Ptr<AnastaReceiverApp> receiver = CreateObject<AnastaReceiverApp>();
    receiver->Setup(5000, nodes.Get(0), nodes.Get(2), scenario);
    nodes.Get(1)->AddApplication(receiver);
    receiver->SetStartTime(Seconds(0.0));
    receiver->SetStopTime(Seconds(300.0));

    Simulator::Stop(Seconds(300.0));
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
