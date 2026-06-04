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

            // Fetch the base MobilityModel safely
            Ptr<MobilityModel> uavMob = m_uavNode->GetObject<MobilityModel>();
            Ptr<MobilityModel> jammerMob = m_jammerNode->GetObject<MobilityModel>();
            double distance = uavMob->GetDistanceFrom(jammerMob);
	    if (pSize > 14 && distance < 30.0) {
                double dropProbability = 100.0 * (1.0 - (distance / 30.0) * (distance / 30.0));
                
                // 1. SIMULATE ERASURE
                if ((rand() % 100) < dropProbability) {
                    std::cout << "[ns-3] 💥 JAMMER DESTROYED PACKET! (Dist: " << distance << "m | Drop Prob: " << dropProbability << "%)" << std::endl;
                    delete[] buffer;
                    continue; 
                }
                
                // 2. SIMULATE CORRUPTION: Packet survived, but payload gets corrupted
                uint32_t corruptIdx = 14 + (rand() % (pSize - 14));
                buffer[corruptIdx] ^= 0xFF; 
                std::cout << "[ns-3] ⚡ JAMMER CORRUPTED PACKET! Distance: " << distance << "m." << std::endl;
            } else {
                std::cout << "[ns-3] ✅ Packet passed cleanly. Distance: " << distance << "m." << std::endl;
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
    
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer devices = wifi.Install(phy, mac, nodes);

    MobilityHelper mobility;
    
    // Set Base Station (Node 1) and Jammer (Node 2) to fixed positions at 50m
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes.Get(1));
    mobility.Install(nodes.Get(2));
    nodes.Get(1)->GetObject<MobilityModel>()->SetPosition(Vector(50.0, 0.0, 0.0));
    nodes.Get(2)->GetObject<MobilityModel>()->SetPosition(Vector(50.0, 0.0, 0.0));

    // Set UAV (Node 0) to a Waypoint Patrol Path (Flies back and forth at 1 m/s)
    MobilityHelper uavMobHelper;
    uavMobHelper.SetMobilityModel("ns3::WaypointMobilityModel");
    uavMobHelper.Install(nodes.Get(0));
    Ptr<WaypointMobilityModel> uavMob = nodes.Get(0)->GetObject<WaypointMobilityModel>();

    // Waypoint 1: Start at 0m (t=0s)
    uavMob->AddWaypoint(Waypoint(Seconds(0.0), Vector(0.0, 0.0, 0.0)));
    // Waypoint 2: Fly to 100m (Passes through Jammer at t=50s)
    uavMob->AddWaypoint(Waypoint(Seconds(100.0), Vector(100.0, 0.0, 0.0)));
    // Waypoint 3: Turn around and fly back to 0m (Passes Jammer again at t=150s)
    uavMob->AddWaypoint(Waypoint(Seconds(200.0), Vector(0.0, 0.0, 0.0)));
    // Waypoint 4: Fly forward to 100m again (Passes Jammer a third time at t=250s)
    uavMob->AddWaypoint(Waypoint(Seconds(300.0), Vector(100.0, 0.0, 0.0)));

    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    address.Assign(devices);

    Ptr<AnastaSenderApp> sender = CreateObject<AnastaSenderApp>();
    sender->Setup(9000, Ipv4Address("10.1.1.2"), 80);
    nodes.Get(0)->AddApplication(sender);
    sender->SetStartTime(Seconds(0.0));
    sender->SetStopTime(Seconds(300.0));

    Ptr<AnastaReceiverApp> receiver = CreateObject<AnastaReceiverApp>();
    receiver->Setup(5000, nodes.Get(0), nodes.Get(2));
    nodes.Get(1)->AddApplication(receiver);
    receiver->SetStartTime(Seconds(0.0));
    receiver->SetStopTime(Seconds(300.0));

    std::cout << "🚀 Real-Time Network Simulation Active. Duration: 300s" << std::endl;
    Simulator::Stop(Seconds(300.0)); 
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}