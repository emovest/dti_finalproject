from flask import Flask, request, jsonify, session
from recommender import predict, recommend_paper

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # 若使用 session，必须设置

@app.route('/webhook', methods=['POST'])  # Dialogflow 会发送请求到这个路径
def webhook():
    data = request.get_json(silent=True)
    user_input = data.get("queryResult", {}).get("queryText", "")

    if not user_input:
        return jsonify({"fulfillmentText": "No user input received."}), 200

    try:
        final_label = predict(user_input)
        paper = recommend_paper(user_input)

        # 可选：保存 label 以备后用
        session['final_label'] = final_label

        if paper is None:
            return jsonify({
                "fulfillmentText": f"Predicted label: {final_label}. No matching paper found."
            }), 200

        return jsonify({
            "fulfillmentText": (
                f"📌 Recommended Paper: \n\n"
                f"📄 Title:\n{paper['original_title'].values[0]}\n\n"
                f"— — — — —\n"
                f"📝 Abstract:\n\n"
                f"{paper['original_abstract'].values[0]}\n\n"
            )
        })


    except Exception as e:
        return jsonify({
            "fulfillmentText": f"An error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)


    
