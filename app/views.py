from flask import jsonify, render_template, request

from app import app
from app.core import present_result, present_result_filtered

# http://flask.pocoo.org/docs/1.0/
VERSION = "0.0.4"


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        msg = request.form.get("msg")
        print(msg)
        app.logger.info("Pergunta: {}".format(msg))
        answer = present_result(msg)

        print(answer)
        app.logger.info("Resposta: {}".format(answer))

        details = []
        for key, value in answer["metadata"].items():
            details.append(
                value["source"].split("/")[-1] + ": Página " + str(value["page"])
            )
        return render_template(
            "index.html", answer=answer["response"], details=details, msg=msg
        )
    return render_template(
        "index.html",
    )


@app.route("/demo", methods=["GET", "POST"])
def demo():
    if request.method == "POST":
        msg = request.form.get("msg")
        print(request.form)
        medication = request.form.get("medicamento")
        print(msg)
        strength = request.form.get("dosagem")

        app.logger.info("Pergunta: {}".format(msg))
        answer = present_result_filtered(msg, medication, strength)

        print(answer)
        app.logger.info("Resposta: {}".format(answer))

        details = []
        for key, value in answer["metadata"].items():
            details.append(
                value["source"].split("/")[-1] + ": Página " + str(value["page"])
            )
        return render_template(
            "demo.html",
            answer=answer["response"],
            details=details,
            msg=msg,
            selected_medicamento=medication,
        )
    return render_template(
        "demo.html",
    )


@app.route("/demochat", methods=["GET", "POST"])
def demochat():
    if request.method == "POST":
        data = request.get_json()  # Automatically parses JSON
        print(data)
        msg = data.get("user_message")
        # print(request.form)
        medication = data.get("medicamento")
        # print(msg)
        strength = data.get("dosagem")
        recent_history = data.get("recent_history", [])  # List of recent messages
        # if recent_history:
        #    msg = "\n".join(
        #        f"{entry['sender']}: {entry['message']}" for entry in recent_history
        #    )
        app.logger.info("Pergunta: {}".format(msg))
        answer = present_result_filtered(msg, medication, strength)

        print(answer)
        app.logger.info("Resposta: {}".format(answer))

        details = []
        for key, value in answer["metadata"].items():
            details.append(
                value["source"].split("/")[-1] + ": Página " + str(value["page"])
            )
        details_better = "Documentos usados para procurar a resposta \n" + "\n".join(
            details
        )
        return jsonify(
            {"response": answer["response"], "additional_info": details_better}
        )

    return render_template(
        "demochat.html",
    )


@app.route("/evaluate", methods=["POST"])
def eval():
    # print("aqui")
    # Aqui você acessa os dados do formulário
    hipotese_escolhida = request.form.get("hipotese")
    pergunta = request.form.get("pergunta")
    resposta = request.form.get("resposta")

    print("Hipótese escolhida:", hipotese_escolhida)
    app.logger.info(
        "##EVAL --> PERGUNTA: {}||RESPOSTA: {}||Avaliação: {}".format(
            pergunta, resposta, hipotese_escolhida
        )
    )

    return render_template("index.html")


@app.route("/api/v1/results", methods=["POST"])
def apiv1():
    msg = request.form.get("msg")
    print(request.form)
    medication = request.form.get("medicamento")
    print(msg)
    strength = request.form.get("dosagem")

    app.logger.info("Pergunta: {}".format(msg))
    answer = present_result_filtered(msg, medication, strength)

    print(answer)
    app.logger.info("Resposta: {}".format(answer))

    details = []
    for key, value in answer["metadata"].items():
        details.append(
            value["source"].split("/")[-1] + ": Página " + str(value["page"])
        )
    return jsonify(
        {"message": answer["message"], "details": details, "resumo": answer["resumo"]}
    )
