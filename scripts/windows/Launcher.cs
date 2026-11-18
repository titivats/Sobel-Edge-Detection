using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Windows.Forms;

internal static class Launcher
{
    private const string PasswordVariable = "AVTR_SETTINGS_PASSWORD";

    [STAThread]
    private static int Main(string[] args)
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        try
        {
            string root = AppDomain.CurrentDomain.BaseDirectory;
            string runtime = Path.Combine(root, "_runtime");
            string python = Path.Combine(runtime, "pythonw.exe");
            string app = Path.Combine(root, "production_app.py");
            if (!File.Exists(python) || !File.Exists(app))
                throw new FileNotFoundException("ไฟล์โปรแกรมไม่ครบ กรุณาติดตั้ง AVTR อีกครั้ง หรือคัดลอกโฟลเดอร์ AVTR ทั้งชุดพร้อม _runtime และ production_app.py");

            string password = Environment.GetEnvironmentVariable(PasswordVariable);
            if (String.IsNullOrWhiteSpace(password))
                password = LocalPassword(Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "AVTR"));

            var arguments = new StringBuilder("-E -s -B ");
            arguments.Append(QuoteArgument(app));
            foreach (string arg in args)
                arguments.Append(' ').Append(QuoteArgument(arg));
            var start = new ProcessStartInfo(python, arguments.ToString())
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = root,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            start.EnvironmentVariables["TORCH_HOME"] = Path.Combine(runtime, "torch");
            start.EnvironmentVariables[PasswordVariable] = password ?? "";

            var tail = new Queue<string>();
            DataReceivedEventHandler receive = delegate(object sender, DataReceivedEventArgs line)
            {
                if (String.IsNullOrEmpty(line.Data)) return;
                lock (tail)
                {
                    if (tail.Count == 16) tail.Dequeue();
                    tail.Enqueue(line.Data.Length > 500 ? line.Data.Substring(0, 500) : line.Data);
                }
            };
            using (var child = new Process { StartInfo = start })
            {
                child.OutputDataReceived += receive;
                child.ErrorDataReceived += receive;
                child.Start();
                child.BeginOutputReadLine();
                child.BeginErrorReadLine();
                child.WaitForExit();
                if (child.ExitCode != 0)
                {
                    string detail = String.Join(Environment.NewLine, tail.ToArray());
                    if (!String.IsNullOrEmpty(password)) detail = detail.Replace(password, "***");
                    MessageBox.Show("AVTR ปิดด้วยรหัส " + child.ExitCode +
                        (detail.Length > 0 ? Environment.NewLine + detail : ""),
                        "AVTR", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                return child.ExitCode;
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show(ex.Message, "เปิด AVTR ไม่สำเร็จ", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }

    // Windows passes argv as one string. Backslashes before a quote, including
    // the closing quote, must be escaped using the C runtime's argument rules.
    internal static string QuoteArgument(string value)
    {
        var result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char c in value)
        {
            if (c == '\\') { slashes++; continue; }
            if (c == '"') result.Append('\\', slashes * 2 + 1).Append(c);
            else result.Append('\\', slashes).Append(c);
            slashes = 0;
        }
        return result.Append('\\', slashes * 2).Append('"').ToString();
    }

    private static string LocalPassword(string folder)
    {
        string path = Path.Combine(folder, "settings-password.bin");
        if (File.Exists(path))
        {
            try { return ReadPassword(path); }
            catch (Exception ex)
            {
                MessageBox.Show("อ่านรหัส SETTING ของผู้ใช้ Windows นี้ไม่ได้ ระบบจะเปิดหน้าแสดงผล\n" +
                    "คุณสามารถกำหนด AVTR_SETTINGS_PASSWORD ก่อนเปิดใหม่\n\n" + ex.Message,
                    "AVTR", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return null;
            }
        }

        string password = CreatePassword();
        if (password == null) return null;
        string temporary = Path.Combine(folder, Guid.NewGuid().ToString("N") + ".tmp");
        try
        {
            Directory.CreateDirectory(folder);
            byte[] plain = Encoding.UTF8.GetBytes(password);
            byte[] encrypted;
            try { encrypted = ProtectedData.Protect(plain, null, DataProtectionScope.CurrentUser); }
            finally { Array.Clear(plain, 0, plain.Length); }
            using (var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                stream.Write(encrypted, 0, encrypted.Length);
                stream.Flush(true);
            }
            try { File.Move(temporary, path); }
            catch (IOException)
            {
                if (!File.Exists(path)) throw;
                return ReadPassword(path);
            }
            return password;
        }
        catch (Exception ex)
        {
            MessageBox.Show("บันทึกรหัส SETTING ไม่สำเร็จ ใช้รหัสที่เพิ่งตั้งได้เฉพาะครั้งนี้\n\n" + ex.Message,
                "AVTR", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return password;
        }
        finally
        {
            try { File.Delete(temporary); }
            catch (IOException) {}
            catch (UnauthorizedAccessException) {}
        }
    }

    private static string ReadPassword(string path)
    {
        byte[] plain = ProtectedData.Unprotect(File.ReadAllBytes(path), null, DataProtectionScope.CurrentUser);
        try { return Encoding.UTF8.GetString(plain); }
        finally { Array.Clear(plain, 0, plain.Length); }
    }

    private static string CreatePassword()
    {
        using (var dialog = new Form())
        {
            dialog.Text = "AVTR · ตั้งรหัส SETTING ครั้งแรก";
            dialog.StartPosition = FormStartPosition.CenterScreen;
            dialog.FormBorderStyle = FormBorderStyle.FixedDialog;
            dialog.MinimizeBox = false;
            dialog.MaximizeBox = false;
            dialog.AutoScaleMode = AutoScaleMode.Dpi;
            dialog.Font = SystemFonts.MessageBoxFont;
            dialog.AutoSize = true;
            dialog.AutoSizeMode = AutoSizeMode.GrowAndShrink;
            var fields = new TableLayoutPanel { AutoSize = true, Dock = DockStyle.Fill, ColumnCount = 2,
                Padding = new Padding(16) };
            dialog.Controls.Add(fields);
            var note = new Label { Text = "กำหนดรหัสเพื่อเลือกแหล่งข้อมูลและแก้ไข SETTING\nรหัสนี้เก็บไว้เฉพาะบัญชี Windows นี้",
                AutoSize = true, Margin = new Padding(3, 3, 3, 14) };
            fields.Controls.Add(note, 0, 0);
            fields.SetColumnSpan(note, 2);
            var first = new TextBox { Width = 240, UseSystemPasswordChar = true };
            var second = new TextBox { Width = 240, UseSystemPasswordChar = true };
            fields.Controls.Add(new Label { Text = "รหัส SETTING", AutoSize = true, Anchor = AnchorStyles.Left }, 0, 1);
            fields.Controls.Add(first, 1, 1);
            fields.Controls.Add(new Label { Text = "ยืนยันรหัส", AutoSize = true, Anchor = AnchorStyles.Left }, 0, 2);
            fields.Controls.Add(second, 1, 2);
            var buttons = new FlowLayoutPanel { AutoSize = true, FlowDirection = FlowDirection.RightToLeft, Dock = DockStyle.Fill };
            var save = new Button { Text = "บันทึกและเปิด", AutoSize = true };
            var skip = new Button { Text = "เปิดหน้าแสดงผล", AutoSize = true, DialogResult = DialogResult.Cancel };
            save.Click += delegate
            {
                if (first.Text.Trim().Length == 0 || first.Text != second.Text || first.Text != first.Text.Trim())
                {
                    MessageBox.Show(dialog, "กรอกรหัสให้ตรงกันทั้งสองช่อง โดยเว้นว่างไม่ได้และไม่มีช่องว่างที่หัวหรือท้าย",
                        "รหัส SETTING", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return;
                }
                dialog.DialogResult = DialogResult.OK;
            };
            buttons.Controls.Add(save);
            buttons.Controls.Add(skip);
            fields.Controls.Add(buttons, 0, 3);
            fields.SetColumnSpan(buttons, 2);
            dialog.AcceptButton = save;
            dialog.CancelButton = skip;
            return dialog.ShowDialog() == DialogResult.OK ? first.Text : null;
        }
    }
}
