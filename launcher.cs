using System;
using System.Diagnostics;
using System.IO;

class Program
{
    static void Main()
    {
        string exeDir = Path.GetDirectoryName(
            System.Reflection.Assembly.GetExecutingAssembly().Location);

        var psi = new ProcessStartInfo
        {
            FileName = "uv",
            Arguments = "run --project \"" + exeDir + "\" vsf-gui",
            WorkingDirectory = exeDir,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        try
        {
            Process.Start(psi);
        }
        catch (Exception ex)
        {
            System.Windows.Forms.MessageBox.Show(
                "起動に失敗しました。uv がインストールされているか確認してください。\n\n" + ex.Message,
                "vsf-gui",
                System.Windows.Forms.MessageBoxButtons.OK,
                System.Windows.Forms.MessageBoxIcon.Error);
        }
    }
}
