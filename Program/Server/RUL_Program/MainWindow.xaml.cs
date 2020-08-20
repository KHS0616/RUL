using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace RUL_Program
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        // 서버 시작 버튼
        private void btnStartServer_Click(object sender, RoutedEventArgs e)
        {
            Server.StartMqttServer();
            Server.StartMobiusServer();
            Server.ConnectMqttServer("192.168.219.192");
        }

        // 서버 종료 버튼
        private void btnEndServer_Click(object sender, RoutedEventArgs e)
        {
            Server.processMOBIUS.Close();
        }

        // 게이트웨이 이동 버튼
        private void btnConnect_Click(object sender, RoutedEventArgs e)
        {
            // 이벤트를 발생시킨 버튼의 정보 저장
            Button btn = sender as Button;
            
            // 버튼의 Content에 따른 게이트웨이 생성
            int index = Array.FindIndex(Server.gateWaysNames, i => i == btn.Content.ToString());
            Server.gateWays[index] = new GateWay(btn.Content.ToString());

            // 새 창에서 게이트웨이 관련 창 열기
            Window window = Server.gateWays[index];
            window.Show();
            Server.SubscribeMqttTopic("Mobius/Gyeyang");
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            Server.ConnectMqttServer("192.168.0.10");
/*            Server.CreateClient();
            Server.PostProduct("1");*/
        }
    }
}
