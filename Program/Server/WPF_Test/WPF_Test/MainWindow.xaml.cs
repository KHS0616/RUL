using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Diagnostics;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;
using WPF_Test.Properties;
using ConsoleLib;

namespace WPF_Test
{
    public partial class MainWindow : Window
    {
        // 프로그램 실행 시, 첫 화면
        // 서버 정보 불러오기
        public MainWindow()
        {
            InitializeComponent();

            // 소켓 및 IP정보 불러오기
            Server.BeginServer();

        }

        // 서버 연결 버튼 클릭 이벤트
        private void Button_Click_ServerCreate(object sender, RoutedEventArgs e)
        {
            if(btnServerConnect.Content.ToString() == "서버 시작")
            {
                Server.ConnectServer();
                Mobius.CreateClient();
                btnServerConnect.IsEnabled = false;
            }
            else
            {

            }
        }


        // 명령 전송 버튼 클릭 이벤트
        private void SendText(object sender, RoutedEventArgs e)
        {
            // 전송할 문자열 변수 선언
            string text = "";

            // 입력받은 버튼에 따른 명령 분기
            switch(sender.ToString().Substring(32))
            {
                case "BandPass":
                    text = "bandpass";
                    Server.SendText(text);
                    break;
                case "RowData":
                    text = "row";
                    Server.SendText(text);
                    break;
                case "FindPeak":
                    text = "peak";
                    Server.SendText(text);
                    break;
            }

            // 전송 완료 후 텍스트박스에 추가하고, 원래의 내용은 지운다.
            AppendText(string.Format("[보냄]{0}: {1}", Server.thisAddress.ToString(), text));
        }

        // 데이터 송수신 로그 화면에 텍스트 출력 메소드
        public void AppendText(string content)
        {
            // 외부 스레드에서 컨트롤 접근시 발생하는 에러 방지
            Dispatcher.Invoke(DispatcherPriority.Normal, new Action(delegate
            {
                TextLog.AppendText(content + '\n');
            }));            
        }

        // 화면 전환 이벤트
        private void btnSensor001_Click(object sender, RoutedEventArgs e)
        {            
            switch (sender.ToString().Substring(32))
            {
                case "계양역":
                    Server.gateWays[0] = new GateWay("계양역");
                    Window sensor001 = Server.gateWays[0];
                    sensor001.Show();
                    break;
                default:
                    NodeSqlFormatter("");
                    break;
            }
            
        }

        // NodeJS 실행 메소드
        // 모비우스 서버 실행을 위한 메소드
        private string NodeSqlFormatter(string sourceText)
        {
            ProcessStartInfo cmd = new ProcessStartInfo();
            Process process = new Process();

            cmd.FileName = @"cmd";  // cmd 창 실행
            cmd.WindowStyle = ProcessWindowStyle.Hidden; //cmd 창 숨기기
            cmd.CreateNoWindow = false;      //cmd 창 띄우지 않기
            cmd.UseShellExecute = false;   // Shell 기능 미사용
            cmd.RedirectStandardInput = true;  // cmd 창에서 데이터를 가져오기
            cmd.RedirectStandardOutput = true; // cmd 창으로 데이터 보내기
            cmd.RedirectStandardError = true; //cmd 창에서 오류 내용 가져오기

            process.StartInfo = cmd;
            process.EnableRaisingEvents = false;   // 종료 이벤트 알리지 않기
            process.Start();
            process.StandardInput.Write(@"cd ../../Mobius-master" + Environment.NewLine);
            process.StandardInput.Write(@"node mobius.js" + Environment.NewLine);
            //process.StandardInput.Write(@"ipconfig" + Environment.NewLine);
            process.StandardInput.Close();
            StreamReader reader = process.StandardOutput;
            string s = reader.ReadToEnd();
            s = s.Replace("\n", "\r\n");
            MessageBox.Show(s);
            //process.WaitForExit();
            //process.Close();
            //process.Close();
            /*            Process p = new Process();

                        // 경로 구분자 설정
                        string path = Directory.GetParent(Environment.CurrentDirectory).Parent.FullName;

                        string s = "";

                        p.StartInfo.UseShellExecute = false;
                        p.StartInfo.RedirectStandardInput = true;
                        p.StartInfo.RedirectStandardOutput = true;
                        p.StartInfo.StandardOutputEncoding = Encoding.UTF8;
                        p.StartInfo.WindowStyle = ProcessWindowStyle.Normal;
                        p.StartInfo.CreateNoWindow = false;

                        p.StartInfo.FileName = string.Format("C:\\Program Files\\nodejs\\node.exe");

                        p.StartInfo.Arguments = string.Format(@"{0}\Mobius-master\mobius.js", path);
                        p.Start();

                        StreamReader reader = p.StandardOutput;
                        StreamWriter writer = p.StandardInput;
                        writer.AutoFlush = true;
                        s = reader.ReadToEnd();
                        s = s.Replace("\n", "\r\n");*/

            return "";
        }
    }
}
