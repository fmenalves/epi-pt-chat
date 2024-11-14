from flask import render_template, request

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


# Rinvoq
# Ozempic
# Paclitaxel Accord
# Diovan
# Influvac
# Comirnaty - create
# Triticum
# Depakine
# Lenalidomida Tecnigen
# Ciplox - create


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
            "demo.html", answer=answer["response"], details=details, msg=msg
        )
    return render_template(
        "demo.html",
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
