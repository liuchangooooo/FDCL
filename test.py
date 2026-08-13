from openai import OpenAI
import os

# 推荐从环境变量读取密钥，避免硬编码泄露
# 使用前请先在系统环境变量中配置 OPENAI_API_KEY
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    # 若使用中转/代理服务，可在此添加 base_url="你的代理地址"
    base_url="https://ai.lv10.ren/v1"
)

def test_openai_api():
    try:
        # 调用聊天补全接口
        response = client.chat.completions.create(
            model="gpt-5.5",
            messages=[
                {"role": "system", "content": "你是一个简洁的测试助手。"},
                {"role": "user", "content": "请回复：OpenAI 接口测试成功"}
            ],
            temperature=0.7
        )

        # 提取并打印模型回复
        reply_content = response.choices[0].message.content
        print("✅ 调用成功，模型回复：")
        print(reply_content)

        # 打印 Token 消耗统计
        usage = response.usage
        print(f"\n📊 Token消耗：输入 {usage.prompt_tokens}，输出 {usage.completion_tokens}，总计 {usage.total_tokens}")
        return True

    except Exception as e:
        print(f"❌ 调用失败，错误信息：{str(e)}")
        return False

if __name__ == "__main__":
    test_openai_api()
