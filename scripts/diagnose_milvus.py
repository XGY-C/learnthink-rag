"""Milvus 知识库诊断工具 - 检查数据状态和测试查询"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.retrievers.milvus_retriever import (
    connect,
    get_or_create_collection,
    retrieve,
)
from app.embedding import encode_query, encode_query_sparse, encode_query_hybrid


def check_collection_info(course_id: str):
    """检查 Collection 的基本信息"""
    print(f"\n{'='*60}")
    print(f"检查课程: {course_id}")
    print(f"{'='*60}")
    
    try:
        col = get_or_create_collection(course_id)
        
        # 基本信息
        print(f"\n✓ Collection 名称: {col.name}")
        print(f"✓ 记录总数: {col.num_entities}")
        print(f"✓ Schema 字段:")
        for field in col.schema.fields:
            print(f"  - {field.name}: {field.dtype} (dim={field.params.get('dim', 'N/A')})")
        
        # 索引信息
        print(f"\n✓ 索引信息:")
        for index in col.indexes:
            print(f"  - 字段: {index.field_name}")
            print(f"    类型: {index.params.get('index_type', 'N/A')}")
            print(f"    度量: {index.params.get('metric_type', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        return False


def test_search_modes(course_id: str, query: str = "什么是搜索算法"):
    """测试不同的搜索模式"""
    print(f"\n{'='*60}")
    print(f"测试查询: '{query}'")
    print(f"{'='*60}")
    
    modes = ["dense", "sparse", "hybrid"]
    
    for mode in modes:
        print(f"\n--- 测试 {mode.upper()} 模式 ---")
        try:
            results = retrieve(
                course_id=course_id,
                query=query,
                k=5,
                search_mode=mode,
            )
            
            print(f"✓ 返回结果数: {len(results)}")
            if results:
                print(f"\n前 3 个结果:")
                for i, r in enumerate(results[:3], 1):
                    print(f"  {i}. [{r['relevance']:.4f}] {r['doc_title']}")
                    print(f"     摘要: {r['excerpt'][:100]}...")
            else:
                print("⚠ 未找到任何匹配结果")
                
        except Exception as e:
            print(f"✗ 错误: {e}")


def test_topic_filter(course_id: str, query: str = "贝叶斯", topic: str = "ch03_贝叶斯分类器"):
    """测试带 topic 过滤的查询"""
    print(f"\n{'='*60}")
    print(f"测试 Topic 过滤")
    print(f"{'='*60}")
    print(f"查询: '{query}'")
    print(f"Topic: '{topic}'")
    
    try:
        results = retrieve(
            course_id=course_id,
            query=query,
            k=5,
            topic=topic,
        )
        
        print(f"\n✓ 返回结果数: {len(results)}")
        if results:
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['relevance']:.4f}] {r['doc_title']}")
        else:
            print("⚠ 未找到任何匹配结果（可能该 topic 下没有相关内容）")
            
    except Exception as e:
        print(f"✗ 错误: {e}")


def check_vector_encoding(query: str = "测试查询"):
    """检查向量编码是否正常"""
    print(f"\n{'='*60}")
    print(f"测试向量编码")
    print(f"{'='*60}")
    print(f"查询文本: '{query}'")
    
    try:
        # Dense 向量
        dense_vec = encode_query(query)
        print(f"\n✓ Dense 向量维度: {len(dense_vec)}")
        print(f"  前 5 个值: {[f'{v:.4f}' for v in dense_vec[:5]]}")
        
        # Sparse 向量
        sparse_vec = encode_query_sparse(query)
        print(f"\n✓ Sparse 向量非零元素数: {len(sparse_vec)}")
        if sparse_vec:
            sample_items = list(sparse_vec.items())[:3]
            print(f"  示例: {[(k, f'{v:.4f}') for k, v in sample_items]}")
        
        # Hybrid
        dense_h, sparse_h = encode_query_hybrid(query)
        print(f"\n✓ Hybrid 编码成功")
        print(f"  Dense: {len(dense_h)} 维")
        print(f"  Sparse: {len(sparse_h)} 非零元素")
        
        return True
    except Exception as e:
        print(f"\n✗ 向量编码失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Milvus 知识库诊断工具")
    print("="*60)
    
    # 连接 Milvus
    print("\n[1/5] 连接 Milvus...")
    try:
        connect()
        print("✓ 连接成功")
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return
    
    # 检查向量编码
    print("\n[2/5] 测试向量编码...")
    check_vector_encoding("人工智能基础")
    
    # 检查 Collection 信息
    course_id = "course-ai-001"
    print(f"\n[3/5] 检查 Collection 信息...")
    if not check_collection_info(course_id):
        print("\n⚠ Collection 不存在或无法访问，请先运行 build_index.py")
        return
    
    # 测试不同搜索模式
    print(f"\n[4/5] 测试搜索模式...")
    test_search_modes(course_id, "什么是搜索算法")
    
    # 测试 topic 过滤
    print(f"\n[5/5] 测试 Topic 过滤...")
    test_topic_filter(course_id, "贝叶斯", "ch03_贝叶斯分类器")
    
    print("\n" + "="*60)
    print("诊断完成！")
    print("="*60)


if __name__ == "__main__":
    main()
