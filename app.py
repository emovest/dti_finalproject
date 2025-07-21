from flask import Flask, request, jsonify
from upstash_redis import Redis
import os
from recommend_more import recommend_more_from_liked_paper
from recommender import predict, recommend_paper

app = Flask(__name__)

# 初始化 Upstash Redis（确保 Render 设置了这两个环境变量）
redis = Redis(
    url=os.environ.get("UPSTASH_REDIS_REST_URL"),
    token=os.environ.get("UPSTASH_REDIS_REST_TOKEN")
)

@app.route('/')
def home():
    return "✅ Server is running!"

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()

    # Dialogflow 的结构中，intent name 是在 queryResult 中
    intent = data["queryResult"]["intent"]["displayName"]
    user_input = data["queryResult"]["queryText"]
    user_id = data["session"]  # 可以简化处理

    print(f"🎯 Received intent: {intent}")

    # 如果是主推荐意图
    if intent == "getUserCrytoInterest":
        result = recommender(user_input)

        # 把推荐中的第一篇的文本和标签存入 Redis
        liked_text = result[0]["paper"]
        liked_label = result[0]["label"]
        redis.set(f"{user_id}:liked_text", liked_text)
        redis.set(f"{user_id}:liked_label", liked_label)

        # 返回推荐结果（这里只返回第一篇）
        return jsonify({
            "fulfillmentMessages": [
                {"text": {"text": [f"📄 Here's a paper: {result[0]['title']}"]}}
            ]
        })

    # 如果是请求更多推荐的意图
    elif intent == "getUserIntentforMorePaper":
        liked_text = redis.get(f"{user_id}:liked_text")
        liked_label = redis.get(f"{user_id}:liked_label")

        if liked_text is None or liked_label is None:
            return jsonify({
                "fulfillmentMessages": [
                    {"text": {"text": ["⚠️ Sorry, I couldn't find your previous preferences. Please tell me your research interest again."]}}
                ]
            })

        more_results = recommend_more_from_liked_paper(liked_text, liked_label)
        response_text = "📚 Here are more papers:\n" + "\n".join([p["title"] for p in more_results])

        return jsonify({
            "fulfillmentMessages": [
                {"text": {"text": [response_text]}}
            ]
        })

    # 兜底情况
    else:
        return jsonify({
            "fulfillmentMessages": [
                {"text": {"text": ["❓ I didn't quite understand that."]}}
            ]
        })

if __name__ == '__main__':
    app.run(debug=True)

    
