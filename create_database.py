#!/usr/bin/env python3
"""创建 Milvus 数据库"""

from pymilvus import MilvusClient

# 连接到 Milvus
client = MilvusClient(uri="http://localhost:19530")

# 创建数据库
db_name = "learnthink"
try:
    client.create_database(db_name)
    print(f"✅ 数据库 '{db_name}' 创建成功")
except Exception as e:
    if "already exist" in str(e).lower():
        print(f"ℹ️  数据库 '{db_name}' 已存在")
    else:
        print(f"❌ 创建数据库失败: {e}")
