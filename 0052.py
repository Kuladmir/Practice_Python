import pymysql as pm
con = pm.connect(host="localhost",user="root",password="",db="jxgl",charset="utf8")
cuiobj = con.cursor()
cuiobj.execute("Select * From specialty")
row = cuiobj.fetchone()
print(row)