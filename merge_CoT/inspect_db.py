import chromadb

DB_PATH = "./reflexion_full_db"

def inspect_rules():
    print(f"📂 正在打开数据库: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = client.get_collection(name="rule_book")
    except:
        print("❌ 数据库或集合不存在，请先运行训练脚本。")
        return

    count = collection.count()
    print(f"📊 当前存储的经验总数: {count}")
    
    if count == 0:
        return

    # 获取所有数据
    data = collection.get()
    
    print("\n=== 经验列表 (前 20 条) ===")
    for i in range(min(count, 20)):
        rid = data['ids'][i]
        pattern = data['metadatas'][i].get('pattern', 'N/A')
        strategy = data['documents'][i]
        print(f"ID: {rid}")
        print(f"📌 Trigger (Pattern): {pattern}")
        print(f"💡 Strategy (Logic):  {strategy}")
        print("-" * 50)

if __name__ == "__main__":
    inspect_rules()