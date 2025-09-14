import subprocess
import sys
import os

# 统一的运行函数
def run_script(script_name):
    print(f"\n🚀 开始执行 {script_name} ...\n")
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"❌ {script_name} 执行失败，退出码 {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n✅ {script_name} 执行完成！\n")


def main():
    scripts = [
        #"extract_frame.py",          # 第1步：提取第120秒帧
        "run_sam_batch.py",          # 第2步：SAM分割试纸条
        "redlines.py",               # 第3步：红线定位
        "export_redline_videos.py",  # 第4步：输出红线区域视频（前120s）
    ]

    for s in scripts:
        if not os.path.exists(s):
            print(f"⚠️ 脚本 {s} 不存在，跳过")
            continue
        run_script(s)

    print("\n🎉 全流程执行完毕！请检查 output 目录：")
    print("  - image/                  ← 第1步帧图像")
    print("  - crops/                  ← 第2步试纸条 + strip_coords.json")
    print("  - redlines/               ← 第3步红线静图 + redline_coords.json")
    print("  - redline_videos/         ← 第4步红线区域视频（前120秒）")


if __name__ == "__main__":
    main()
