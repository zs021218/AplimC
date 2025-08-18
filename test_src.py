#!/usr/bin/env python3
"""
测试 src 模块功能
"""

import src

# 打印横幅
src.print_banner()

print("\n📋 项目信息:")
project_info = src.get_project_info()
for key, value in project_info.items():
    print(f"  {key}: {value}")

print("\n🔧 版本信息:")
version_info = src.get_version_info()
for key, value in version_info.items():
    print(f"  {key}: {value}")

print("\n🌍 环境信息:")
env_info = src.check_environment()
for key, value in env_info.items():
    print(f"  {key}: {value}")

print("\n✅ src 模块测试完成!")
