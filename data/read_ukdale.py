import pandas as pd 
from pandas import HDFStore
from pprint import pprint
import matplotlib.pyplot as plt
plt.ion()

Building1 = {
             "name": "building1",
             "ID":1,
             "apps" : ["dish washer","fridge","kettle","microwave","washing machine"],
             "main_meter":1,
             "meters":[6,12,10,13,5],
             "start": "2013-01-01",
             "end": "2017-01-01"} 

Building2 = {"name": "building2",
             "ID":2,
             "apps" : ["dish washer","fridge","kettle","microwave","washing machine"],
             "main_meter":1,
             "meters":[13,14,8,15,12],
             "start": "2013-03-01",
             "end": "2013-10-01"} 

Building5 = { "name": "building5",
              "ID":5,
             "apps" : ["dish washer","fridge","kettle","microwave","washing machine"],
             "main_meter":1,
             "meters":[22,19,18,23,24],
             "start": "2014-07-01",
             "end": "2014-11-01"} 

BUILDINGS =[Building1,Building2,Building5]

def read_ukdale(path= "ukdale.h5",out="uk_dale.h5"):
    # Leer datos edificio para las fechas indicadas
    all_bldngs = []
    for bldng in BUILDINGS:
        app_data =[]
        with HDFStore('ukdale.h5') as data_store:
            for ii,meter in enumerate(bldng["meters"]):
                print(f"Leyendo casa {bldng['ID']} electrodoméstico {bldng['apps'][ii]}")
                appliance_df = data_store.get(f"/{bldng['name']}/elec/meter{meter}")
                appliance_df.columns = [bldng["apps"][ii]]
                app_data.append(appliance_df)
            main = data_store.get(f"/{bldng['name']}/elec/meter{1}")
            main.columns = ['main']
        
        # Filtrar por fecha inicio y final
        print(f"Filtrando la información por fechas {bldng['start']} <--> {bldng['end']} ")
        main = main.loc[bldng["start"]:bldng["end"]]
        app_data = [app.loc[bldng["start"]:bldng["end"]] for app in app_data]        
        # Remuestrear a 6 segundos. Hacer un reindex
        print(f"Remuestreando a 6s")
        date_idx = pd.date_range(start=bldng["start"],end=bldng["end"],freq='6s',tz="Europe/London")
        main = main.reindex(date_idx,method='nearest')
        app_data = [app.reindex(date_idx,method='nearest') for app in app_data]        
        # Concatenar main y las apps
        print(f"Concatenando los datos de la casa {bldng['apps'][ii]} ")
        dataBldng = pd.concat([main]+app_data,axis=1)
        # Concatener el id de la casa
        dataBldng["House"]=bldng['ID']
        all_bldngs.append(dataBldng)
    # Guardar en el archivo pasado como parámetro
    print("Concatenando la información de todas las casas")
    dataUKDALE = pd.concat(all_bldngs,axis=0) 
    print(f"Guardando los datos en {out}")
    dataUKDALE.to_hdf(out,key='df',index=True)
    return dataUKDALE

# Unit testing
if __name__ == "__main__":
    data = read_ukdale(path='ukdale.h5',out="UKDALE.h5") 
    # Test
    #data.loc["2016-01"].plot(subplots=False)
    data = pd.read_hdf('UKDALE.h5')