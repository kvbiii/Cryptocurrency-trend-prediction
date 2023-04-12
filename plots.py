from requirements import *
def plotting(results_łączne, kwartyl_łączne, results_base, name, nazwa_estymatora, results_pojedynczy=None, kwartyl_pojedynczy=None):
    fig = go.Figure()
    i = 0
    traces = []
    X = [i for i in range(len(results_łączne[0])+len(results_łączne[1])+len(results_łączne[2]))]
    labels = ["Fold 1" if index < len(results_łączne[0]) else "Fold 2" if index < len(results_łączne[0]) + len(results_łączne[1]) else "Fold 3" for index in X]
    if(nazwa_estymatora != "LSTM"):
        traces.append(go.Scatter(x=X, y=results_pojedynczy[0].flatten().tolist()+results_pojedynczy[1].flatten().tolist()+results_pojedynczy[2].flatten().tolist(), mode='lines',line=dict(color="green", width=2), text=labels, showlegend=True, name=f"Pojedynczy: Kwartyl {kwartyl_pojedynczy}", xaxis="x"))
    traces.append(go.Scatter(x=X, y=results_łączne[0].flatten().tolist()+results_łączne[1].flatten().tolist()+results_łączne[2].flatten().tolist(), mode='lines',line=dict(color="blue", width=2), text=labels, showlegend=True, name=f"Łączny: Kwartyl {kwartyl_łączne} ", xaxis="x"))
    traces.append(go.Scatter(x=X, y=results_base[0].flatten().tolist()+results_base[1].flatten().tolist()+results_base[2].flatten().tolist(), mode='lines', line=dict(color="black", width=2), text=labels, showlegend=True, name="Model bazowy", xaxis="x2"))
    layout = go.Layout(xaxis=dict(ticks='outside', tickvals=[len(results_łączne[0].flatten().tolist()), len(results_łączne[0].flatten().tolist())+len(results_łączne[1].flatten().tolist())], ticklen = 35, tickwidth=3, showticklabels=False), xaxis2=dict(ticks='outside', tickvals=[i+len(results_łączne[0])//2 for i in [0, len(results_łączne[0]), len(results_łączne[0]) + len(results_łączne[1])]], ticktext=["Fold 1", "Fold 2", "Fold 3"], tickangle=0,ticklen=0,tickwidth=0, showticklabels=True, title="Podzbiór", overlaying="x"), yaxis=dict(nticks=10, tickprefix="$", title="Wartość portfela"))
    fig = go.Figure(data=traces, layout=layout)
    if(len(name.split('_')) > 1):
        title_krypto = name.split('_')[0].capitalize()+"*"
    else:
        title_krypto = name.split('_')[0].capitalize()
    fig.update_layout(title_text='<b>{} {}</b>'.format(nazwa_estymatora, title_krypto), title_x=0.5, template="simple_white", width=1000, height=600, showlegend=True, legend=dict(yanchor="top", y=1.1, xanchor="left", x=0.01), font=dict(family="Times New Roman",size=20,color="Black"))
    fig.show(renderer="browser")