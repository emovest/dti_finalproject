from flask import Flask, request, jsonify, session
from recommender import predict, recommend_paper

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # 若使用 session，必须设置

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json(silent=True)
    user_input = data.get("queryResult", {}).get("queryText", "")
    intent_name = data.get("queryResult", {}).get("intent", {}).get("displayName", "")

    if not user_input:
        return jsonify({"fulfillmentText": "No user input received."}), 200

    try:
        # === Intent 1: 用户初次请求推荐论文 ===
        if intent_name == "getUserCrytoInterest":
            final_label = predict(user_input)
            paper = recommend_paper(user_input)

            if paper is None:
                return jsonify({
                    "fulfillmentText": f"Predicted label: {final_label}. No matching paper found."
                }), 200

            # 存下这次推荐的论文（作为用户喜欢的）
            session["liked_text"] = paper["paper"].values[0]
            session["final_label"] = final_label

            return jsonify({
                "fulfillmentText": (
                    f"📌 Recommended Paper: \n\n"
                    f"📄 Title:\n{paper['original_title'].values[0]}\n\n"
                    f"— — — — —\n"
                    f"📝 Abstract:\n\n"
                    f"{paper['original_abstract'].values[0]}\n\n"
                )
            })

        # === Intent 2: 用户想要更多推荐 ===
        elif intent_name == "getUserIntentforMorePaper":
            liked_text = session.get("liked_text")
            label = session.get("final_label")

            if not liked_text or not label:
                return jsonify({
                    "fulfillmentText": "Sorry, I couldn't find your previous paper preference."
                }), 200

            more_papers = recommend_more_from_liked_paper(liked_text, label, top_k=5)

            if not more_papers or len(more_papers) == 0:
                return jsonify({
                    "fulfillmentText": "No additional similar papers found at the moment."
                }), 200

            # 拼接5篇论文的摘要与标题
            response_text = "📚 Here are some more papers you might like:\n\n"
            for idx, row in enumerate(more_papers.itertuples(), 1):
                response_text += (
                    f"🔹 Paper {idx}:\n"
                    f"📄 Title: {row.original_title}\n"
                    f"📝 Abstract: {row.original_abstract}\n"
                    f"— — — — —\n\n"
                )

            return jsonify({"fulfillmentText": response_text})

        # === Default fallback ===
        else:
            return jsonify({"fulfillmentText": "Sorry, I didn't understand that."}), 200

    except Exception as e:
        return jsonify({
            "fulfillmentText": f"An error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
    
