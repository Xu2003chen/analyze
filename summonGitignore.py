"""
生成一个 .gitignore 文件，内容为当前目录及其子目录下所有大于 x 的文件。
"""

import os
import sys

# 定义文件大小阈值
SIZE_THRESHOLD_MB = 200
SIZE_THRESHOLD_BYTES = SIZE_THRESHOLD_MB * 1024 * 1024

# 用于存储找到的大文件路径
large_files = []

# 获取脚本运行的当前目录
current_dir = os.getcwd()
print(f"正在搜索目录: {current_dir} (及其所有子目录)")

# 使用 os.walk 递归遍历所有子目录
# root: 当前遍历到的目录路径
# dirs: 当前目录下的子目录列表
# files: 当前目录下的文件列表
for root, dirs, files in os.walk(current_dir):
    for file in files:
        # 构造文件的完整路径
        file_path = os.path.join(root, file)

        try:
            # 获取文件大小
            file_size = os.path.getsize(file_path)

            # 检查文件大小是否超过阈值
            if file_size > SIZE_THRESHOLD_BYTES:
                # 计算相对于 current_dir 的路径，以便写入 .gitignore
                # os.path.relpath(path, start) 返回 path 相对于 start 的路径
                relative_path = os.path.relpath(file_path, current_dir)
                print(f"找到大文件 ({file_size / (1024*1024):.2f} MB): {relative_path}")
                large_files.append(relative_path)
        except OSError as e:
            # 处理可能的权限错误或文件已被删除等情况
            print(f"警告: 无法访问文件 {file_path}: {e}")

# 将找到的大文件路径写入 .gitignore
gitignore_path = os.path.join(current_dir, ".gitignore")

# 检查 .gitignore 是否已存在
if os.path.exists(gitignore_path):
    print(f"\n警告: .gitignore 文件已存在 ({gitignore_path})")
    user_input = input("是否要追加内容到现有文件? (y/N): ")
    if user_input.lower() != "y":
        print("操作已取消。")
        sys.exit(0)
    write_mode = "a"  # 追加模式
    print("将以追加模式写入...")
else:
    write_mode = "w"  # 写入模式
    print(f"\n正在创建 .gitignore 文件: {gitignore_path}")

try:
    with open(gitignore_path, write_mode) as f:
        if write_mode == "a" and os.path.getsize(gitignore_path) > 0:
            # 如果是追加且文件不为空，先加个换行
            f.write("\n")
        if large_files:
            for file_path in large_files:
                # 确保路径使用正斜杠，这在 Windows 和 Unix 上都兼容
                normalized_path = file_path.replace("\\", "/")
                f.write(f"{normalized_path}\n")
            print(f"\n成功将 {len(large_files)} 个大文件路径写入 .gitignore。")
        else:
            print(f"\n未找到大于 {SIZE_THRESHOLD_MB}MB 的文件。")

except IOError as e:
    print(f"错误: 无法写入 .gitignore 文件: {e}")
    sys.exit(1)

print("\n操作完成。")
