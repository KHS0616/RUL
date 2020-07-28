using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using WPF_Test.Properties;
using WPF_Test;
using Newtonsoft.Json.Linq;
using System.Windows.Threading;

namespace WPF_Test
{
    class Server 
    {
        // 소켓 변수 선언
        public static Socket mainSock;
        public static IPAddress thisAddress;
        // 클라이언트 가변 리스트 선언
        public static List<Socket> connectedClients = new List<Socket>();

        // 폼 제어를 위한 변수 선언
        public static MainWindow mainWindow = ((MainWindow)System.Windows.Application.Current.MainWindow);

        // 게이트웨이 정보 저장
        public static GateWay[] gateWays = new GateWay[10];
        public static string[] gateWayName = {"계양역", "계산역" };

        // 서버 연결 이전 기본 정보 저장(주소, 소켓 등)
        public static void BeginServer()
        {
            // 소켓을 생성하고 소켓 정보를 저장
            Server.mainSock = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.IP);
            IPHostEntry he = Dns.GetHostEntry(Dns.GetHostName());
            // 처음으로 발견되는 ipv4 주소를 사용한다.
            foreach (IPAddress addr in he.AddressList)
            {
                if (addr.AddressFamily == AddressFamily.InterNetwork)
                {
                    Server.thisAddress = addr;
                    break;
                }
            }
            // 주소가 없다면..
            if (Server.thisAddress == null)
                // 로컬호스트 주소를 사용한다.
                Server.thisAddress = IPAddress.Loopback;
            mainWindow.txtAddress.Text = Server.thisAddress.ToString();
        }

        // 서버 연결
        public static void ConnectServer()
        {
            // 통신을 위한 쓰레드 생성
            // 쓰레드 생성 및 실행
            int port;
            if (!int.TryParse(mainWindow.txtPort.Text, out port))
            {
                MessageBox.Show("포트 번호가 잘못 입력되었거나 입력되지 않았습니다.");
                mainWindow.txtPort.Focus();
                mainWindow.txtPort.SelectAll();
                return;
            }
            // 서버에서 클라이언트의 연결 요청을 대기하기 위해
            // 소켓을 열어둔다.
            IPEndPoint serverEP = new IPEndPoint(Server.thisAddress, port);
            Server.mainSock.Bind(serverEP);
            Server.mainSock.Listen(10);

            // 비동기적으로 클라이언트의 연결 요청을 받는다.
            mainWindow.TextLog.AppendText("서버가 시작되었습니다.\n");
            //Server_Connect_Button.Content = "서버 종료";
            Server.mainSock.BeginAccept(Server.AcceptCallback, null);
        }

        //클라이언트 연결 콜백 메소드
        // 클라이언트의 연결요청 수락
        public static void AcceptCallback(IAsyncResult ar)
        {
            // 클라이언트의 연결 요청을 수락한다.
            Socket client = Server.mainSock.EndAccept(ar);

            // 또 다른 클라이언트의 연결을 대기한다.
            Server.mainSock.BeginAccept(AcceptCallback, null);
            AsyncObject obj = new AsyncObject(400000);
            obj.WorkingSocket = client;

            // 연결된 클라이언트 리스트에 추가해준다.
            Server.connectedClients.Add(client);

            // 서버 로그에 클라이언트가 연결되었다고 써준다.
            mainWindow.AppendText(string.Format("클라이언트 (@ {0})가 연결되었습니다.", client.RemoteEndPoint));

            // 클라이언트의 데이터를 받는다.
            client.BeginReceive(obj.Buffer, 0, 400000, 0, Server.DataReceived, obj);
        }

        //데이터 수신
        public static void DataReceived(IAsyncResult ar)
        {
            // BeginReceive에서 추가적으로 넘어온 데이터를 AsyncObject 형식으로 변환한다.
            AsyncObject obj = (AsyncObject)ar.AsyncState;

            // 데이터 수신을 끝낸다.
            int received = obj.WorkingSocket.EndReceive(ar);

            // 받은 데이터가 없으면(연결끊어짐) 끝낸다.
            if (received <= 0)
            {
                obj.WorkingSocket.Close();
                return;
            }

            // 텍스트로 변환한다.
            var text = Encoding.UTF8.GetString(obj.Buffer).Trim('\0');
            ProcessingJson(text);

            // 데이터를 받은 후엔 다시 버퍼를 비워주고 같은 방법으로 수신을 대기한다.
            obj.ClearBuffer();

            // 수신 대기
            obj.WorkingSocket.BeginReceive(obj.Buffer, 0, 400000, 0, DataReceived, obj);
        }

        // JSON 명령 수행
        public static void ProcessingJson(string text)
        {
            JObject msg = JObject.Parse(text);
            string type = msg["Type"].ToString();
            string name = msg["Name"].ToString();
            string content = msg["Content"].ToString();
            string data = msg["Data"].ToString();
            string time = msg["Time"].ToString();
            string comment = msg["Comment"].ToString();
            int findIndex = Array.FindIndex(Server.gateWayName, i => i == name.Split()[0]);

            
            mainWindow.AppendText(string.Format("[받음]:{0}", comment));
            if(type.Equals("Step") || type.Equals("RS"))
            {
                switch (content)
                {
                    case "Sin":
                        Server.gateWays[findIndex].sensors[Server.gateWays[findIndex].thisPage - 1].Dispatcher.Invoke(DispatcherPriority.Normal, new Action(delegate
                        {
                            Server.gateWays[findIndex].sensors[Server.gateWays[findIndex].thisPage - 1].lbl_Sin.Content = data;
                            Server.gateWays[findIndex].sensors[Server.gateWays[findIndex].thisPage - 1].btn_Sin.IsEnabled = true;
                        }));
                        break;
                    case "Raw":
                    case "Bandpass":
                    case "Peak":
                        Server.gateWays[findIndex].sensors[Server.gateWays[findIndex].thisPage - 1].byteArrayToImage(data);
                        break;
                    case "Load Spec Setting":
                        string[] tempData = new string[4];
                        tempData = data.Split();
                        Server.gateWays[findIndex].sensors[Server.gateWays[findIndex].thisPage - 1].Step_Fs = tempData[0];
                        Server.gateWays[findIndex].sensors[Server.gateWays[findIndex].thisPage - 1].Step_Time = tempData[1];
                        Server.gateWays[findIndex].sensors[Server.gateWays[findIndex].thisPage - 1].Step_DifStep = tempData[2];
                        Server.gateWays[findIndex].sensors[Server.gateWays[findIndex].thisPage - 1].Step_PerSpec = tempData[3];                        
                        break;
                }
            }
            else if(type.Equals("AD"))
            {
                switch(content)
                {
                    case "String":
                        Mobius.PostProduct(data);
                        if(!comment.Split()[2].Equals("Normal"))
                        {
                            MessageBox.Show(comment);
                        }
                        break;
                }
            }

        }

        // 서버 -> 클라이언트 문자열 데이터 전송
        public static void SendText(string content)
        {
            // 서버가 대기중인지 확인한다.
            if (!Server.mainSock.IsBound)
            {
                MessageBox.Show("서버가 실행되고 있지 않습니다!");
                return;
            }

            // 문자열을 utf8 형식의 바이트로 변환한다.
            byte[] bDts = Encoding.UTF8.GetBytes(content);
            // 연결된 모든 클라이언트에게 전송한다.
            for (int i = Server.connectedClients.Count - 1; i >= 0; i--)
            {
                Socket socket = Server.connectedClients[i];
                try { socket.Send(bDts); }
                catch
                {
                    // 오류 발생하면 전송 취소하고 리스트에서 삭제한다.
                    try { socket.Dispose(); } catch { }
                    Server.connectedClients.RemoveAt(i);
                }
            }
        }
    }
}
