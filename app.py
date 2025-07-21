from flask import Flask, request, jsonify
from recommender import predict, recommend_paper  # 确保文件名就是 recommder.py，没有拼错

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json(silent=True)
    
    if not data or "user_input" not in data:
        return jsonify({"error": "No user input provided"}), 400

    user_input = data["user_input"]

    try:
        final_label = predict(user_input)
        paper = recommend_paper(user_input)

        if paper is None:
            return jsonify({
                "label": final_label,
                "message": "No paper found for this label."
            }), 200

        return jsonify({
            "label": final_label,
            "recommended_title": paper["original_title"].values[0],
            "recommended_abstract": paper["original_abstract"].values[0]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

    