#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h> 

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("AnastaSimulationBridge");

// 1. SENDER APPLICATION (Installed on UAV - Node 0)
class AnastaSenderApp : public Application {
public:
    AnastaSenderApp() : m_hostListenFd(-1), m_running(false) {}
    virtual ~AnastaSenderApp() {}

    void Setup(uint16_t listenPort, Ipv4Address destIp, uint16_t destPort) {
        m_listenPort = listenPort;
        m_destIp = destIp;
        m_destPort = destPort;

        m_hostListenFd = socket(AF_INET, SOCK_DGRAM, 0);
        int flags = fcntl(m_hostListenFd, F_GETFL, 0);
        fcntl(m_hostListenFd, F_SETFL, flags | O_NONBLOCK);

        sockaddr_in listenAddr{};
        listenAddr.sin_family = AF_INET;
        listenAddr.sin_port = htons(m_listenPort);
        listenAddr.sin_addr.s_addr = INADDR_ANY;
        bind(m_hostListenFd, (struct sockaddr*)&listenAddr, sizeof(listenAddr));
    }

protected:
    virtual void StartApplication(void) override {
        m_running = true;
        m_ns3Socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_ns3Socket->Connect(InetSocketAddress(m_destIp, m_destPort));
        Simulator::Schedule(MilliSeconds(10), &AnastaSenderApp::PollHostSocket, this);
    }

    virtual void StopApplication(void) override {
        m_running = false;
        if (m_hostListenFd >= 0) close(m_hostListenFd);
    }

private:
    void PollHostSocket() {
        if (!m_running) return;

        uint8_t buffer[2048];
        ssize_t bytesRecv = recv(m_hostListenFd, buffer, sizeof(buffer), 0);
        
        if (bytesRecv > 0) {
            std::cout << "[ns-3 Ingest] Captured " << bytesRecv << " bytes from edge_node.py. Injecting into Wi-Fi..." << std::endl;
            Ptr<Packet> packet = Create<Packet>(buffer, bytesRecv);
            m_ns3Socket->Send(packet);
        }
        Simulator::Schedule(MilliSeconds(10), &AnastaSenderApp::PollHostSocket, this);
    }

    int m_hostListenFd;
    uint16_t m_listenPort;
    Ipv4Address m_destIp;
    uint16_t m_destPort;
    bool m_running;
    Ptr<Socket> m_ns3Socket;
};


// 2. RECEIVER APPLICATION (Installed on Base Station - Node 1)
class AnastaReceiverApp : public Application {
public:
    AnastaReceiverApp() : m_hostSendTwinFd(-1) {}
    virtual ~AnastaReceiverApp() {}

    void Setup(uint16_t twinPort, Ptr<Node> uavNode, Ptr<Node> jammerNode) {
        m_twinPort = twinPort;
        m_uavNode = uavNode;
        m_jammerNode = jammerNode;

        m_hostSendTwinFd = socket(AF_INET, SOCK_DGRAM, 0);
        m_twinAddr.sin_family = AF_INET;
        m_twinAddr.sin_port = htons(m_twinPort);
        m_twinAddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    }

protected:
    virtual void StartApplication(void) override {
        m_ns3Socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_ns3Socket->Bind(InetSocketAddress(Ipv4Address::GetAny(), 80));
        m_ns3Socket->SetRecvCallback(MakeCallback(&AnastaReceiverApp::ReceiveFromNs3Channel, this));
    }

    virtual void StopApplication(void) override {
        if (m_hostSendTwinFd >= 0) close(m_hostSendTwinFd);
    }

private:
    void ReceiveFromNs3Channel(Ptr<Socket> socket) {
        Ptr<Packet> packet;
        Address from;
        while ((packet = socket->RecvFrom(from))) {
            uint32_t pSize = packet->GetSize();
            uint8_t* buffer = new uint8_t[pSize];
            packet->CopyData(buffer, pSize);

            Ptr<MobilityModel> uavMob = m_uavNode->GetObject<MobilityModel>();
            Ptr<MobilityModel> jammerMob = m_jammerNode->GetObject<MobilityModel>();
            double distance = uavMob->GetDistanceFrom(jammerMob);

            // Jammer activates only when UAV flies within 15 meters of the Base Station coordinate space
            if (pSize > 14 && distance < 15.0) {
                
                if (rand() % 100 < 80) {
                    std::cout << "[ns-3 Egest] 💥 JAMMER DESTROYED PACKET! Signal lost at " << distance << "m." << std::endl;
                    delete[] buffer;
                    continue; // Skip forwarding to the Python Twin
                }
                
                uint32_t corruptIdx = 14 + (rand() % (pSize - 14));
                buffer[corruptIdx] ^= 0xFF; 
                std::cout << "[ns-3 Egest] ⚡ JAMMER CORRUPTED PACKET! Distance: " << distance << "m. Forwarding..." << std::endl;
            } else {
                std::cout << "[ns-3 Egest] ✅ Packet passed cleanly. Distance: " << distance << "m." << std::endl;
            }

            sendto(m_hostSendTwinFd, buffer, pSize, 0, (struct sockaddr*)&m_twinAddr, sizeof(m_twinAddr));
            delete[] buffer;
        }
    }

    int m_hostSendTwinFd;
    uint16_t m_twinPort;
    sockaddr_in m_twinAddr;
    Ptr<Socket> m_ns3Socket;
    Ptr<Node> m_uavNode;
    Ptr<Node> m_jammerNode;
};

int main(int argc, char *argv[]) {
    GlobalValue::Bind ("SimulatorImplementationType", StringValue ("ns3::RealtimeSimulatorImpl"));
    CommandLine cmd;
    cmd.Parse(argc, argv);

    NodeContainer nodes;
    nodes.Create(3); 

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    
    // FIX: Default helper already installs LogDistancePropagationLossModel.
    // Do not chain AddPropagationLoss a second time to prevent double attenuation.
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer devices = wifi.Install(phy, mac, nodes);

    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
    mobility.Install(nodes);

    // UAV setup: Starts at 0m, flies forward smoothly at 1.0 m/s
    Ptr<ConstantVelocityMobilityModel> uavMob = nodes.Get(0)->GetObject<ConstantVelocityMobilityModel>();
    uavMob->SetPosition(Vector(0.0, 0.0, 0.0));
    uavMob->SetVelocity(Vector(1.0, 0.0, 0.0)); 

    // Base Station setup: Positioned cleanly at 50m
    Ptr<ConstantVelocityMobilityModel> bsMob = nodes.Get(1)->GetObject<ConstantVelocityMobilityModel>();
    bsMob->SetPosition(Vector(50.0, 0.0, 0.0));
    bsMob->SetVelocity(Vector(0.0, 0.0, 0.0));

    // Jammer setup: Positioned at 50m (co-located next to the critical asset target)
    Ptr<ConstantVelocityMobilityModel> jammerMob = nodes.Get(2)->GetObject<ConstantVelocityMobilityModel>();
    jammerMob->SetPosition(Vector(50.0, 0.0, 0.0));
    jammerMob->SetVelocity(Vector(0.0, 0.0, 0.0));

    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    address.Assign(devices);

    Ptr<AnastaSenderApp> sender = CreateObject<AnastaSenderApp>();
    sender->Setup(9000, Ipv4Address("10.1.1.2"), 80);
    nodes.Get(0)->AddApplication(sender);
    sender->SetStartTime(Seconds(0.0));
    sender->SetStopTime(Seconds(100.0));

    Ptr<AnastaReceiverApp> receiver = CreateObject<AnastaReceiverApp>();
    receiver->Setup(5000, nodes.Get(0), nodes.Get(2));
    nodes.Get(1)->AddApplication(receiver);
    receiver->SetStartTime(Seconds(0.0));
    receiver->SetStopTime(Seconds(100.0));

    std::cout << "🚀 Real-Time Network Simulation Active." << std::endl;
    Simulator::Stop(Seconds(100.0)); // Fixed human timing window extension
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}

