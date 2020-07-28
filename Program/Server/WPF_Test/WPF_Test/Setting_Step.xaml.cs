using Newtonsoft.Json.Linq;
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

namespace WPF_Test
{
    public partial class Setting_Step : Window
    {
        // JSON 변수 선언
        JObject content = new JObject();
        public Setting_Step(string name)
        {
            InitializeComponent();

            // JSON 초기화
            content.Add("Type", "");
            content.Add("Name", name);
            content.Add("Content", "");
            content.Add("Data", "");
            content.Add("Time", "");
            content.Add("Comment", "");

            // 스펙 정보 초기화
            int findIndex = Array.FindIndex(Server.gateWayName, i => i == name.Split()[0]);
            txt_Fs.Text = Server.gateWays[findIndex].sensors[Server.gateWays[findIndex].thisPage - 1].Step_Fs;
            txt_Time.Text = Server.gateWays[findIndex].sensors[Server.gateWays[findIndex].thisPage - 1].Step_Time;
            txt_DifStep.Text = Server.gateWays[findIndex].sensors[Server.gateWays[findIndex].thisPage - 1].Step_DifStep;
            txt_PerSec.Text = Server.gateWays[findIndex].sensors[Server.gateWays[findIndex].thisPage - 1].Step_PerSpec;
        }

        // 스펙 정보 저장 메소드
        private void btn_Setting_Save_Click(object sender, RoutedEventArgs e)
        {
            Double Fs = Double.Parse(txt_Fs.Text);
            int time = Int32.Parse(txt_Time.Text);
            Double DifStep = Double.Parse(txt_DifStep.Text);
            Double PerSec = Double.Parse(txt_PerSec.Text);

            content["Type"] = "Step";
            content["Content"] = "Save Spec Setting";
            content["Data"] = string.Format("{0} {1} {2} {3}", Fs.ToString(), time.ToString(), DifStep.ToString(), PerSec.ToString());
            content["Time"] = System.DateTime.Now.ToString();
            content["Comment"] = "";
            Server.SendText(content.ToString());
            Window.GetWindow(this).Close();
        }

        private void btn_Setting_Cancel_Click(object sender, RoutedEventArgs e)
        {
            Window.GetWindow(this).Close();
        }
    }
}
