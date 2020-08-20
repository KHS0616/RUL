using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;
using uPLibrary.Networking.M2Mqtt;
using uPLibrary.Networking.M2Mqtt.Messages;

namespace RUL_Program
{
    class Server
    {
        #region 모비우스 관련 변수
        // MQTT 정보를 저장할 변수
        public static MqttClient client;

        // 모비우스 서버 프로세스 변수
        public static ProcessStartInfo cmd;
        public static Process processMQTT;
        public static Process processMOBIUS;

        // 모비우스 REST API 전송을 위한 변수 선언
        public static HttpClient clientHTTP = new HttpClient();
        #endregion
        #region 게이트웨이 관련 변수
        // 각 게이트웨이 정보를 관리할 변수
        public static GateWay[] gateWays = new GateWay[5];
        public static string[] gateWaysNames = {"계양역", "계산역"};

        // 메인 원도우 관리 변수
        public static MainWindow mainWindow = ((MainWindow)System.Windows.Application.Current.MainWindow);
        #endregion

        #region 모비우스 접속 관련 메소드
        // 모비우스 서버 실행 메소드
        // cmd 명령어를 통한 MQTT 서버 실행
        public static void StartMqttServer()
        {
            // 커맨드 창과 해당 명령을 실행할 프로세스 할당
            cmd = new ProcessStartInfo();
            processMQTT = new Process();

            // 커맨드 창 설정
            cmd.FileName = @"cmd";  // cmd 창 실행
            cmd.WindowStyle = ProcessWindowStyle.Hidden; //cmd 창 숨기기
            cmd.CreateNoWindow = false;      //cmd 창 띄우지 않기
            cmd.UseShellExecute = false;   // Shell 기능 미사용
            cmd.RedirectStandardInput = true;  // cmd 창에서 데이터를 가져오기
            cmd.RedirectStandardOutput = true; // cmd 창으로 데이터 보내기
            cmd.RedirectStandardError = true; //cmd 창에서 오류 내용 가져오기

            // 프로세스에 커맨드 창 지정 및 시작
            processMQTT.StartInfo = cmd;
            processMQTT.EnableRaisingEvents = false;   // 종료 이벤트 알리지 않기
            processMQTT.Start();

            // 커맨드 창에 입력할 명령
            // MQTT 실행
            processMQTT.StandardInput.Write(@"cd ../../mosquitto" + Environment.NewLine);
            processMQTT.StandardInput.Write(@"mosquitto" + Environment.NewLine);
            //process.StandardInput.Write(@"ipconfig" + Environment.NewLine);

            // MQTT 실행 후 커맨드 창 및 프로세스 종료
            processMQTT.StandardInput.Close();
            //StreamReader reader = process.StandardOutput;
            //string s = reader.ReadToEnd();
            //s = s.Replace("\n", "\r\n");
            //MessageBox.Show(s);
            processMQTT.WaitForExit();
            processMQTT.Close();
        }

        // cmd 명령어를 통한 모비우스 서버 실행
        public static void StartMobiusServer()
        {
            // 커맨드 창과 해당 명령을 실행할 프로세스 할당
            cmd = new ProcessStartInfo();
            processMOBIUS = new Process();

            // 커맨드 창 설정
            cmd.FileName = @"cmd";  // cmd 창 실행
            cmd.WindowStyle = ProcessWindowStyle.Hidden; //cmd 창 숨기기
            cmd.CreateNoWindow = true;      //cmd 창 띄우지 않기
            cmd.UseShellExecute = false;   // Shell 기능 미사용
            cmd.RedirectStandardInput = true;  // cmd 창에서 데이터를 가져오기
            cmd.RedirectStandardOutput = false; // cmd 창으로 데이터 보내기
            cmd.RedirectStandardError = true; //cmd 창에서 오류 내용 가져오기

            // 프로세스에 커맨드 창 지정 및 시작
            processMOBIUS.StartInfo = cmd;
            processMOBIUS.EnableRaisingEvents = false;   // 종료 이벤트 알리지 않기
            processMOBIUS.Start();

            // 커맨드 창에 입력할 명령
            // MOBIUS 실행
            processMOBIUS.StandardInput.Write(@"cd ../../Mobius-master" + Environment.NewLine);
            processMOBIUS.StandardInput.Write(@"node mobius.js" + Environment.NewLine);

            // MOBIUS실행 후 커맨드 창 종료
            processMOBIUS.StandardInput.Close();
        }

        // MQTT 클라이언트 등록
        public static void ConnectMqttServer(string IPAdress)
        {
            try
            {
                client = new MqttClient(IPAdress);
                client.Connect(Guid.NewGuid().ToString());
                client.MqttMsgPublishReceived += new MqttClient.MqttMsgPublishEventHandler(client_MqttMsgPublishReceived);
            }
            catch
            {
                //label4.Text = "Can't connect to server";
            }
        }
        #endregion

        #region REST API 명령 전송
        // 모비우스 접속을 위한 클라이언트, REST API 헤더 설정
        public static void CreateClient()
        {
            clientHTTP.BaseAddress = new Uri("http://192.168.0.10:7579");
            clientHTTP.DefaultRequestHeaders.Accept.Add(
                new MediaTypeWithQualityHeaderValue("application/json"));
            clientHTTP.DefaultRequestHeaders.Add("X-M2M-RI", "12345");
            clientHTTP.DefaultRequestHeaders.Add("X-M2M-Origin", "{{aei}}");
        }

        // Get API 전송
        public static async void GetProductAsync(string path)
        {
            string test = "";
            clientHTTP.DefaultRequestHeaders.Add("X-M2M-Origin", "{{aei}}");
            HttpResponseMessage response = await clientHTTP.GetAsync(path);
            if (response.IsSuccessStatusCode)
            {
                test = await response.Content.ReadAsStringAsync();
            }
            MessageBox.Show(test);
        }

        // Post API 전송
        public static async void PostProduct(string content)
        {
            try
            {
                // 전송할 API Body 생성 및 Content-Type 설정
                var sendAPI = new StringContent("{\n    \"m2m:cin\": {\n        \"con\": \"" + content + "\"\n    }\n}");
                sendAPI.Headers.ContentType = MediaTypeHeaderValue.Parse("application/vnd.onem2m-res+json; ty=4");

                // API 전송 후, 에러 응답이 있을 경우 오류코드를 받는다.
                var response = await clientHTTP.PostAsync("Hello-Mobius/inhatc-pi/Gateway", sendAPI);
                response.EnsureSuccessStatusCode();
            }
            catch (HttpRequestException ex)
            {
                MessageBox.Show(ex.Message);
            }
            finally
            {

            }
        }
        #endregion

        #region MQTT 데이터 수신
        // MQTT 토픽 Subscribe 등록
        public static void SubscribeMqttTopic(string topic)
        {
            client.Subscribe(new string[] { topic }, new byte[] { (byte)0});
        }

        // MQTT 토픽 Subscribe 등록 해제
        public static void UnSubscribeMqttTopic(string topic)
        {
            client.Unsubscribe(new string[] { topic });
        }

        // MQTT 데이터 수신 이벤트 핸들러
        public static void client_MqttMsgPublishReceived(object sender, MqttMsgPublishEventArgs e)
        {
            string comments = ProcessingJson(System.Text.Encoding.UTF8.GetString(e.Message).Trim('\0'));
            // 서버 로그 창에 기록
            // 외부 스레드에서 컨트롤 접근시 발생하는 에러 방지
            mainWindow.Dispatcher.Invoke(DispatcherPriority.Normal, new Action(delegate
            {
                mainWindow.lblServerLog.Content += string.Format("{0}\r\n", comments);
            }));
        }

        // MQTT JSON 데이터 해석 실행 메소드
        // JSON 명령 수행
        public static string ProcessingJson(string text)
        {
            JObject msg = JObject.Parse(text);
            string type = msg["Type"].ToString();
            string name = msg["Name"].ToString();
            string content = msg["Content"].ToString();
            string data = msg["Data"].ToString();
            string time = msg["Time"].ToString();
            string comment = msg["Comment"].ToString();
            int findIndex = Array.FindIndex(Server.gateWaysNames, i => i == name.Split()[0]);

            // 센서 구분을 위한 임시 리스트
            string[] sensorList = {"1호기", "2호기", "3호기", "4호기" };
            int sensorIndex = Array.FindIndex(sensorList, i => i == name.Split()[1]);

            if (type.Equals("Step") || type.Equals("RS"))
            {
                switch (content)
                {
                    case "Sin":
                        Server.gateWays[findIndex].gatewaySensors[sensorIndex].Dispatcher.Invoke(DispatcherPriority.Normal, new Action(delegate
                        {
                            Server.gateWays[findIndex].gatewaySensors[sensorIndex].lblSin.Content = data;
                        }));
                        break;
                    case "Raw":
                    case "Bandpass":
                    case "Peak":
                        Server.gateWays[findIndex].gatewaySensors[sensorIndex].byteArrayToImage(data);
                        break;
                    case "Load Spec Setting":
/*                        string[] tempData = new string[4];
                        tempData = data.Split();
                        Server.gateWays[findIndex].gatewaySensors[sensorIndex].Step_Fs = tempData[0];
                        Server.gateWays[findIndex].gatewaySensors[sensorIndex].Step_Time = tempData[1];
                        Server.gateWays[findIndex].gatewaySensors[sensorIndex].Step_DifStep = tempData[2];
                        Server.gateWays[findIndex].gatewaySensors[sensorIndex].Step_PerSpec = tempData[3];*/
                        break;
                }
            }
            else if (type.Equals("AD"))
            {
                switch (content)
                {
                    case "Result":
                        MessageBox.Show(data);
/*                        Mobius.PostProduct(data);
                        if (!comment.Split()[2].Equals("Normal"))
                        {
                            MessageBox.Show(comment);
                        }*/
                        break;
                }
            }

            return comment;
        }
        #endregion
    }
}
