import pandas as pd 
import ipdb
from pprint import pprint
import matplotlib.pyplot as plt
plt.ion()



Building1 = {
             "name": "Hosp_redGrupo",
             "submeters" : [ 'CGBT-2.Montante0',
                             'Radiologia1',
                             'RehabilitacionA',
                             'CPD',
                             'Plantas_2-7',
                            ],
             "main_meter":['CGBT-2.Red-Grupo'],
             "start": "2018-03-01 00:00:00",
             "end": "2019-02-28 23:59:00"} 

def read_hosp(path= "Potencias_hosp_simple.csv",out="HOSP.h5"):
    print(f"Leyendo los datos de {path}")
    data = pd.read_csv(path,index_col='Fecha',parse_dates=True)
    # Filtrando main y submeters por medidor y fecha
    main     = data[Building1["main_meter"]].loc[Building1['start']:Building1['end']]
    app_data = data[Building1["submeters"]].loc[Building1['start']:Building1['end']]
    # Concatenación main y submeters
    print(f"Concatenando los datos main {Building1['main_meter']} y los medidores individuales {Building1['submeters']}")
    dataHOSP = pd.concat([main,app_data],axis=1)
    print(f"Guardando los datos en {out}")
    dataHOSP.to_hdf(out,key='df',index=True)
    return dataHOSP

if __name__ == "__main__":
    data = read_hosp("Potencias_hosp_simple.csv",out="HOSP.h5") 

    data = pd.read_hdf("HOSP.h5")

