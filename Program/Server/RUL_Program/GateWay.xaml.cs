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
using System.Windows.Shapes;

namespace RUL_Program
{
    public partial class GateWay : Window
    {
        #region 센서 관련 변수
        public GatewayMainPage gatewayMainPage = new GatewayMainPage();
        public GatewaySensorPage[] gatewaySensors = new GatewaySensorPage[4];
        public Page[] pages = new Page[5];
        public string[] pageList = {"Main", "1호기", "2호기", "3호기", "4호기" };
        #endregion

        public GateWay(string name)
        {
            InitializeComponent();

            // 각 센서 페이지 객체 생성 및 저장
            gatewaySensors[0] = new GatewaySensorPage(string.Format("{0} 1호기", name));
            gatewaySensors[1] = new GatewaySensorPage(string.Format("{0} 2호기", name));
            gatewaySensors[2] = new GatewaySensorPage(string.Format("{0} 3호기", name));
            gatewaySensors[3] = new GatewaySensorPage(string.Format("{0} 4호기", name));

            // 각 센서 및 메인 페이지 정보를 저장
            pages[0] = gatewayMainPage;
            pages[1] = gatewaySensors[0];
            pages[2] = gatewaySensors[1];
            pages[3] = gatewaySensors[2];
            pages[4] = gatewaySensors[3];

            // 초기 페이지를 Main 페이지로 설정
            frContent.Content = pages[0];
        }

        // 센서 이동 버튼
        private void btnGateway_Click(object sender, RoutedEventArgs e)
        {
            // 이벤트를 발생시킨 버튼의 정보 저장 및 페이지 전환
            Button btn = sender as Button;
            int index = Array.FindIndex(pageList, i => i == btn.Content.ToString());
            frContent.Content = pages[index];
        }
    }
}
