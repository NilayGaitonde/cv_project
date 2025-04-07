import pandas as pd
import numpy as np
from time import strftime

def initialize():
    initial_time = strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame(columns=["SAP ID","Attendance","First Seen","Last Seen"])
    return initial_time,df

def add_attendance(sap_id:str,df:pd.DataFrame) -> pd.DataFrame:
    time = strftime("%Y-%m-%d %H:%M:%S")
    if sap_id not in df["SAP ID"].values:
        print(f"Adding {sap_id} to the attendance list")
        new_row = [{"SAP ID": sap_id, "Attendance": 1,"First Seen":time,"Last Seen":time}]
        df = pd.concat([df,pd.DataFrame(new_row)],axis=0).reset_index(drop=True)
    else:
        print(f"{sap_id} already present")
        df.loc[df["SAP ID"] == sap_id, "Last Seen"] = time
    return df

if __name__=="__main__":
    initial_time, df = initialize()
    df = add_attendance("70321019023",df)
    df = add_attendance("70321019022",df)
    df = add_attendance("70321019024",df)
    df = add_attendance("70321019024",df)
    df = add_attendance("70321019024",df)
    df.to_csv(f"attendance_{initial_time}.csv",index=False)
