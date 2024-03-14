from flask import render_template, request

import os
from app import app
from app.core import present_result

# http://flask.pocoo.org/docs/1.0/
VERSION = "0.0.4"


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        msg = request.form.get("msg")
        print(msg)
        enhanced_prompt = (
            msg
            + " "
            + ".Responder em Português de Portugal. Sem nenhuma palavra em português do brasil."
        )
        answer = present_result(enhanced_prompt)
        print(answer)
        details = []
        for key, value in answer["metadata"].items():
            details.append(
                value["source"].split("/")[-1] + ": Página " + str(value["page"])
            )
        return render_template("index.html", answer=answer["response"], details=details)
    return render_template(
        "index.html",
    )
