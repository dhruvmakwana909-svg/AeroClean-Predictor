from flask import Flask, render_template, request, send_file
import pickle, os, datetime
import numpy as np
import pandas as pd

# Load ML model
model_name = "r_regressor.pkl"
model = pickle.load(open(model_name, 'rb'))

app = Flask(__name__)
excel_file = "AQI_Loginfo.xlsx"

@app.route('/')
def home():
    # Load history table + graph if file exists
    tables = None
    graph_path = None

    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
        tables = df.to_html(index=False)
        graph_path = "static/history_graph.png"

        # Create graph
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(df.index, df["Predicted PM 2.5"])  # color not specified
        plt.title("PM 2.5 Prediction Trend")
        plt.xlabel("Prediction Count")
        plt.ylabel("PM 2.5 Value")
        plt.savefig(graph_path)
        plt.close()

    return render_template("index.html", tables=tables, graph=graph_path)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        try:
            t  = float(request.form['Mean Temperature'])
            TM = float(request.form['Maximum Temperature'])
            Tm = float(request.form['Minimum Temperature'])
            SLP = float(request.form['Sea Level Pressure'])
            H = float(request.form['Humidity'])
            VV = float(request.form['Visibility'])
            V = float(request.form['Wind Speed'])
            VM = float(request.form['Max Wind Speed'])
            user = request.form.get("User Name", "Unknown")

            data = np.array([t, TM, Tm, SLP, H, VV, V, VM])
            pred = model.predict(data.reshape(1, -1))[0]

            # Create row with Timestamp + IP + Username
            row = {
                "User Name": user,
                "IP Address": request.remote_addr,
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Mean Temperature": t,
                "Maximum Temperature": TM,
                "Minimum Temperature": Tm,
                "Sea Level Pressure": SLP,
                "Humidity": H,
                "Visibility": VV,
                "Wind Speed": V,
                "Max Wind Speed": VM,
                "Predicted PM 2.5": round(pred, 2)
            }

            row_df = pd.DataFrame([row])

            # Append or create Excel
            if os.path.exists(excel_file):
                old = pd.read_excel(excel_file)
                new = pd.concat([old, row_df], ignore_index=True)
                new.to_excel(excel_file, index=False)
            else:
                row_df.to_excel(excel_file, index=False)

            # Reload everything for UI
            df = pd.read_excel(excel_file)
            table_html = df.to_html(index=False)

            # Generate new graph
            import matplotlib.pyplot as plt
            graph_path = "static/history_graph.png"
            plt.figure()
            plt.plot(df.index, df["Predicted PM 2.5"])
            plt.title("PM 2.5 Prediction Trend")
            plt.xlabel("Prediction Count")
            plt.ylabel("PM 2.5 Value")
            plt.savefig(graph_path)
            plt.close()

            return render_template("index.html",
                                   prediction_text=f"✅ PM 2.5 is: {round(pred,2)}",
                                   tables=table_html,
                                   graph=graph_path)

        except Exception as e:
            return render_template("index.html", prediction_text=f"❌ Input Error!\n({e})")

@app.route('/download')
def download_history():
    if os.path.exists(excel_file):
        return send_file(excel_file, as_attachment=True)
    return "No history file found!"

if __name__ == "__main__":
    app.run(debug=True)
