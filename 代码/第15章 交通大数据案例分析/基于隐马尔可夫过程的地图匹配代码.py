# -*- coding: utf-8 -*-
import os
import math
import six
import psycopg2
import psycopg2.extras
import numpy as np
import pandas as pd
import time
import datetime
import random
import math
import os
import pandas as pd
import numpy as np
from scipy import stats, sparse
import os
import sys
from datetime import datetime
from sqlalchemy import create_engine
from pandas import DataFrame, Series

######################################## PostgreSQL数据库连接函数 ##################################################
# 导入DataFrame进入数据库函数
def DataframetoPgDB(DFdata, tbnm):
    hostInfo = 'localhost' + ':数据库端口号'
    DBengine = create_engine('postgresql://超级管理员名称:超级管理员密码@%s/数据库名称' % hostInfo)
    DFdata.to_sql(tbnm, DBengine, schema='public', if_exists='replace', index=False, chunksize=10000)


# 进行SQL查询操作(返回Dataframe)
def FetchSQLQuery(sqlstr):
    hostInfo = 'localhost' + ':数据库端口号'
    DBengine = create_engine('postgresql://超级管理员名称:超级管理员密码@%s/数据库名称' % hostInfo)
    DBconn = DBengine.connect()
    try:
        return pd.read_sql_query(sqlstr, DBconn)
    except:
        assert not getattr(pd, 'read_sql_table', None)
        TPDBconn = psycopg2.connect(database="数据库名称", user="超级管理员名称", password="超级管理员密码", host="数据库主机号",
                                    port="数据库端口号")
        tmpcur = TPDBconn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        tmpcur.execute(sqlstr)
        colNames = [desc[0] for desc in tmpcur.description]
        results = tmpcur.fetchall()
        return pd.DataFrame([[row[col] for col in colNames] for row in results], columns=colNames)


# 进行SQL查询操作(返回Turple)
def FetchtoTurple(cmd):
    """Execute an SQL command and fetch (fetchall) the result"""
    conn = psycopg2.connect(database="数据库名称", user="超级管理员名称", password="超级管理员密码", host="数据库主机号",
                            port="数据库端口号")  # 这里自己设置
    cur = conn.cursor()
    cur.execute(cmd)
    return cur.fetchall()


# 进行数据库操作(仅操作,不返回)
def RunSQLQuery(sqlstr):
    conn = psycopg2.connect(database="数据库名称", user="超级管理员名称", password="超级管理员密码", host="数据库主机号",
                            port="数据库端口号")  # 这里自己设置
    cur = conn.cursor()
    cur.execute(sqlstr)
    conn.commit()
    
#################################################  似然值函数  ######################################################################
# GPS点距离道路段距离与该道路段为GPS点真正所在道路的概率函数
def distanceLL(distance):  # 存在函数内定义的可变参数 sigma_z为人为定义参数,
    sigma_z = 4.0  # 该参数与GPS噪音有关,文献中多定义为4(Newson and Kummel 2009)
    """和HMM地图匹配算法中提到的惩罚原理相同,使用了Geometric log likelihood function进行惩罚
        这里可以接受numpy或者scala"""
    # 生成一个自由度20,方差为sigma_z为4的似然值(这sigma_z可通过GPS点偏离edge的Median Absolute Deviation求解)
    return stats.t(df=20, scale=sigma_z).logpdf(distance)


# GPS点在候选道路段间转移中标准速度和实际速度比值的概率函数
def temporalLL(travelcostratio):
    """在不同候选道路段间穿越的时间比值的Log likelihood function,其中输入代表了"标准时间/记录时间"这种"标准速度/实际速度"比值
        这里可以接受numpy或者scala"""
    temporal_scale = 0.55  # scale parameter for temporal likelihood
    sigma_t = 0.3  # std dev parameter for temporal likelihood
    # ensures that the two distributions match at 1
    temporalLL_ratio = (
                stats.expon(scale=temporal_scale).logpdf(1) - stats.norm(scale=sigma_t).logpdf(0))  # 计算指数函数和正态函数链接处c值

    if isinstance(travelcostratio, list):
        travelcostratio = np.array(travelcostratio)
    if isinstance(travelcostratio, np.ndarray):
        retvals = stats.expon(scale=temporal_scale).logpdf(travelcostratio)
        retvals[travelcostratio > 1] = (
                    stats.norm(1, scale=sigma_t).logpdf(travelcostratio[travelcostratio > 1]) + temporalLL_ratio)
        return retvals * temporal_weight
    else:  # scalar
        if travelcostratio <= 1:
            return stats.expon(scale=temporal_scale).logpdf(travelcostratio) * temporal_weight
        else:
            return (stats.norm(1, scale=sigma_t).logpdf(travelcostratio) + temporalLL_ratio) * temporal_weight


# GPS点在候选道路段间转移中直线距离和标准距离比值的概率函数
def topologicalLL(distratio):
    """在不同候选道路段间穿越的GPS点间实际距离比候选点间Dijkstra距离比值的log likelihood function"""
    sigma_topol = 0.6  # std dev parameter for topological likelihood
    dr = np.maximum(0,
                    np.array(distratio) - 1)  # distratio can be less than 1 if there is a U-turn, so enforce a minimum
    return stats.t(df=20, scale=sigma_topol).logpdf(dr) * topol_weight


###################################################  剔除速度异常点  #######################################################################
# 匹配预处理工作 —— 删除速度超过120KM/H的瞬时记录点
def getTracesDf(traceTable, numLags=5):  # 获得轨迹数据对应的速度值
    # 参数解释
    # traceTable为轨迹集的名称
    # numLags为往后找多少个GPS点计算速度差

    ###### 约定: 轨迹集中,时间戳列名为tms,GPS单点列名为pt_geom
    """得到轨迹集合中往后推指定numLags的GPS点推算速度,单位km/h"""

    selectClause1 = '''SELECT traceid,ptid,\n'''
    # 注:lagClause1这计算速度为除1000后乘3600
    lagClause1 = ',\n'.join(
        ['ST_Distance(geom, l%(lag)s)/NULLIF(l%(lag)stms-tms,0)/1000*60*60 AS speed%(lag)s' % {'lag': lag} for lag in
         range(1, numLags + 1)])
    selectClause2 = '''\nFROM (SELECT traceid,gid AS ptid, pt_geom AS geom,tms,\n'''
    lagClause2 = ',\n'.join([
        'lag(pt_geom,%(lag)s) OVER (ORDER BY gid DESC) AS l%(lag)s, lag(tms,%(lag)s) OVER (ORDER BY gid DESC) AS l%(lag)stms' % {
            'lag': lag} for lag in range(1, numLags + 1)])
    fromClause = '''\nFROM (SELECT *,1 as traceid FROM %s
                  ''' % traceTable
    cmd = selectClause1 + lagClause1 + selectClause2 + lagClause2 + fromClause + ') AS t1) AS t2;'
    return FetchSQLQuery(cmd)


def fetchPtsToDrop(traceTable, numLags, maxSpeed):  # 寻找要丢弃的轨迹定位点
    df = getTracesDf(traceTable, numLags=5)  # 获得给定推移量的速度表

    df['dropPt'] = False  # will we drop that point?
    ptsToDrop = []  # list of (traceid, ptid) tuples
    for gSize in range(1, numLags + 1):  # loop over groups of 1...5 points
        df['minPt'] = df[df.dropPt == False].groupby('traceid').ptid.min()
        # 最终仅有一列,即指定的ptid;行的组成是df[df.dropPt == False]中traceid列的所有可能值;具体填充的是针对不同traceid的minptid()
        df['maxPt'] = df[df.dropPt == False].groupby('traceid').ptid.max()
        df['lagSpeed'] = df.speed1.shift(gSize * -1)  # 创建lagSpeed列,为speed1列向上移gSize(最后的gSize行用NaN补齐)
        df.loc[
            (df.shift(gSize * -1).traceid != df.traceid), 'lagSpeed'] = np.nan  # 将刚才lagSpeed列向上移gSize形成的最后gSize行改为NaN
        dropMask = ((df.lagSpeed > maxSpeed) & (df.speed1 > maxSpeed)) | (
                (df.ptid == df.minPt + gSize - 1) & (df.speed1 > maxSpeed)) | (
                           (df.lagSpeed > maxSpeed) & (df.ptid == df.maxPt))

        for lag in range(0, gSize):  # make sure we get 'intermediate' points in that group too
            df.loc[dropMask.shift(lag) & (df.shift(lag).traceid == df.traceid), 'dropPt'] = True
            df.loc[dropMask.shift(lag) & (df.shift(lag).traceid == df.traceid), ['speed' + str(s) for s in
                                                                                 range(1, 6)]] = np.nan

        # Don't drop last point in a trace
        df.loc[(df.dropPt) & (df.ptid == df.maxPt), 'dropPt'] = False

        # Move the speeds forward one lag
        for lag in range(0, 5):
            oldCols = ['speed' + str(s) for s in range(lag + 1, 6 - gSize)]
            newCols = ['speed' + str(s) for s in range(lag + 1 + gSize, 6)]
            mask = dropMask.shift(gSize + lag) & (df.shift(gSize + lag).traceid == df.traceid)
            df.loc[mask, oldCols] = df.loc[mask, newCols].values
            df.loc[mask, ['speed' + str(s) for s in range(max(lag + 1, 6 - gSize), 6)]] = np.nan

        ptsToDrop += list(df[df.dropPt].set_index('traceid').ptid.iteritems())
        df = df[df.dropPt == False]

    # Finally, select any 'singletons' where there is one errant point
    # that's not picked up by the lagged values
    ptsToDrop += list(df[df.speed1 > maxSpeed].set_index('traceid').ptid.iteritems())
    return ptsToDrop


def dropPoints(ptsToDrop, traceTable, newtraceTable):
    """Update the database"""
    traceIds = list(set([pt[0] for pt in ptsToDrop]))

    for traceId in traceIds:
        ptIds = [pt[1] for pt in ptsToDrop if pt[0] == traceId]
        dropIds = str(tuple(ptIds)) if len(ptIds) > 1 else '(' + str(ptIds[0]) + ')'
        cmdDict = {'traceTable': traceTable, 'dropIds': dropIds, 'newtraceTable': newtraceTable}
        cmd = '''SELECT * into %(newtraceTable)s FROM %(traceTable)s 
                    WHERE gid NOT IN %(dropIds)s;''' % cmdDict
        RunSQLQuery(cmd)


def reportDroppedPoints(ptsToDrop, traceTable, newtraceTable, logFn,tracenum):
    # 获得车辆信息
    recordnumstr = '''SELECT COUNT(*) FROM %s;''' % traceTable
    orgrecordnum = FetchtoTurple(recordnumstr)[0][0]

    recordnumstr = '''SELECT COUNT(*) FROM %s;''' % newtraceTable
    modrecordnum = FetchtoTurple(recordnumstr)[0][0]
    droprecordnum = orgrecordnum - modrecordnum

    sqlplaterstr = '''SELECT DISTINCT(platenum) FROM %s;''' % traceTable
    platerstr = FetchtoTurple(sqlplaterstr)[0][0]

    with open(logFn, 'a') as f:
        f.write('%s,%s,%s,%s\n' % (platerstr,tracenum, orgrecordnum, droprecordnum))

    # 得到修复好的车辆信息
    getrecordstr = '''SELECT * FROM %s;''' % newtraceTable
    traceGPS = FetchSQLQuery(getrecordstr)
    del traceGPS['pt_geom']

    # 删除对应表格
    droptablestr = '''DROP TABLE %s;DROP TABLE %s''' % (traceTable, newtraceTable)
    RunSQLQuery(droptablestr)

    # 若剩余不到3个点,则返回none;否则进行赋值
    if int(modrecordnum) >= 3:
        return traceGPS
    else:
        return


# 匹配预处理工作 —— 删除速度超过120KM/H的瞬时记录点_______总函数
def DelSpdError(traceGPS, logFn,
                preprocesstbnm,tracenum,newtraceTable):  # 存在函数内定义的可变参数 numLags,maxSpeed,srid,newtraceTable.其中newtraceTable为中间的过程表格
    # 参数区
    numLags = 5  # 计算GPS点速度时往后推几个
    traceTable = preprocesstbnm  # 存储轨迹定位的表格
    maxSpeed = 120  # speed threshold for deleting pings (kmh) in traceCleaner
    srid = '3395'  # 分析使用的投影坐标系SRID
    

    # 向PGSQL输入表格
    DataframetoPgDB(traceGPS, traceTable)
    ### XY转点 —— 暂定二维点
    sqlXYPointstr = '''ALTER TABLE %s ADD COLUMN pt_geom geometry(Point, 4326);
        UPDATE %s SET pt_geom = ST_SetSRID(ST_MakePOINT(lon,lat),4326);''' % (traceTable, traceTable)
    RunSQLQuery(sqlXYPointstr)
    ### 设置SIRD为3395
    sqlTransProj = '''
    ALTER TABLE %s ALTER COLUMN pt_geom TYPE Geometry(Point, 3395) USING ST_Transform(pt_geom,3395)
    ''' % traceTable
    RunSQLQuery(sqlTransProj)

    # 正式运算:浮现fetchPtsToDrop函数
    ptsToDrop = fetchPtsToDrop(traceTable, numLags, maxSpeed)
    if len(ptsToDrop) == 0: # 不需要丢弃任何点,删除traceTable表格
        droptablestr = '''DROP TABLE %s;''' % traceTable
        RunSQLQuery(droptablestr)
        modtraceGPS = traceGPS
        return traceGPS
    else:
        pass
    
    # 生成删除后的轨迹表格
    dropPoints(ptsToDrop, traceTable, newtraceTable)
    # 导出删除超速记录
    modtraceGPS = reportDroppedPoints(ptsToDrop, traceTable, newtraceTable, logFn,tracenum)

    return modtraceGPS


######################################## 计算可能的edge间距离和时间值时,分情况考虑函数 #############################################
def transProbSameEdge(rr1, rr2, dir, sl):
    '''返回的结果将用于edge间移动的likelihood函数,使用该函数的要求是rr1和rr2这两个GPS记录候选candidate为同一个edge
    (rr1.edge==rr1.edge)'''
    frc = abs(rr2['frcalong'] - rr1['frcalong']) if dir == 0 else abs(rr1['frcalong'] - rr2['frcalong'])
    distratio = 1 if sl == 0 else rr1['km'] * frc * 1. / sl
    return max(rr1[['cost', 'reverse_cost'][dir]] * frc * 60 * 60. / max(1, rr2['secs'] - rr1['secs']), 1e-10), max(
        distratio, 1), -1


def transProb(costMatrix, distMatrix, uturncost, rr1, rr2, dir1, dir2, sl):
    # 除了这几个参数之外,还需要输入costMatrix\distMatrix\uturncost(这个需要在循环迭代前就计算出来,为所有道路cost和reversecost的平均值)
    """返回的结果将用于edge间移动的likelihood函数,使用该函数的要求是rr1和rr2这两个GPS记录候选candidate不是同一个edge
       (rr1.edge!=rr1.edge)"""
    if dir1 == 0:
        frc1 = 1 - rr1['frcalong']  # frc of edge remaining
        n0 = rr1['source']
        n1 = rr1['target']
        e1cost = rr1['cost']
    else:
        frc1 = rr1['frcalong']
        n0 = rr1['target']
        n1 = rr1['source']
        e1cost = rr1['reverse_cost']

    if dir2 == 0:
        frc2 = rr2['frcalong']  # frc of edge that will be traveled
        n2 = rr2['source']
        n3 = rr2['target']
        e2cost = rr2['cost']
    else:
        frc2 = 1 - rr2['frcalong']
        n2 = rr2['target']
        n3 = rr2['source']
        e2cost = rr2['reverse_cost']

    # routing cost is cost of 1st edge + routing cost + uturn cost + cost of last edge. e1!=e2 needed in case there are self-loops
    rc = costMatrix[n1, n2]  # routing cost
    uturn = -1
    if rr1['edge'] == rr2['edge'] and dir1 != dir2:
        uturn = 1
        if frc1 > 0.95 and frc2 > 0.95:
            frc1, frc2 = 50, 50  # haven't started the edge and already want to do a U-turn? Disallow
        elif frc1 > frc2:  # continuing along original path, then U turn
            frc1, frc2 = frc1 - frc2, 0
        else:  # U turn, then backtrack
            frc1, frc2 = 0, frc2 - frc1
    elif rr1['edge'] != rr2['edge']:
        e1Uturn = np.round(rc, 8) == np.round(costMatrix[n1, n0] + costMatrix[n0, n2], 8)  # 都保留到8位小数,抹零9位及后面位数
        e2Uturn = np.round(rc, 8) == np.round(costMatrix[n1, n3] + e2cost, 8)
        if e2Uturn:  # U-turn on e2, disallow Uturns on both (causes problems in a grid setting)
            uturn = 2
            e2cost = rr2[['reverse_cost', 'cost'][dir2]]
            frc2 = 1 - frc2
            if frc2 < 0.05: frc2 = 100  # haven't started the edge and already want to do a U-turn? Disallow
        if e1Uturn:  # u-turn on e1
            uturn = 3
            e1cost = rr1[['reverse_cost', 'cost'][dir1]]
            frc1 = 1 - frc1
            if frc1 < 0.05: frc1 = 100  # haven't started the edge and already want to do a U-turn? Disallow
        if e1Uturn and e2Uturn:
            uturn = 4  # note this attracts a greater uturncost
    else:
        somemistake

    if sl == 0 or uturn == 1:  # Uturn on e1, but same edge as e2
        dratio = 1
    else:  # note that frc1 and frc2 have already been changed in the event of a Uturn
        if uturn == -1:  # no Uturn
            networkDist = distMatrix[n1, n2] + rr1['km'] * frc1 + rr2['km'] * frc2
        elif uturn == 2:  # Uturn on e2
            networkDist = distMatrix[n1, n3] + rr1['km'] * frc1 + rr2['km'] * frc2
        elif uturn == 3:  # Uturn on e1
            networkDist = distMatrix[n0, n2] + rr1['km'] * frc1 + rr2['km'] * frc2
        elif uturn == 4:  # Uturn on both
            networkDist = distMatrix[n0, n3] + rr1['km'] * frc1 + rr2['km'] * frc2
        else:
            somemistakehere
        dratio = networkDist * 1. / sl

    cratio = (e1cost * frc1 + rc + uturncost * (uturn > 0) + uturncost * 2 * (
                uturn == 4) + e2cost * frc2) * 60 * 60. / max(1, rr2['secs'] - rr1['secs'])

    # The values will be in a sparse matrix, so zeros aren't distinguishable from empty arrays
    return max(cratio, 1e-10), max(dratio, 1), uturn

###################################################  地图匹配大步骤  ####################################################################
# 第一步骤:计算GPS点同edge间距离
def MapDisToEdge(fwayQuery, tracetbnm, disrticttbnm, streettbnm, weight_1stlast):
    # 参数解释
    # fwayQuery:对于距离放到70米时
    # tracetbnm:车辆轨迹的表格名称
    # disrticttbnm:降低Edge寻找负担的行政区划表,例如town
    # streettbnm:使用的osm2po导入的道路表哥

    starttime = time.time()
    # Step1:寻找每个GPS点可能的edge
    fwayQueryTxt = '' if fwayQuery.strip() == '' else '(' + fwayQuery + ') AND '  # 对高速70m的buffer这种待遇对应的道路进行筛选
    dpStr = '''SELECT * FROM %s''' % tracetbnm

    sqlCandstreet = '''
        SELECT gid::bigint AS path, id, ST_Distance(pt_geom, geom_way),ST_LineLocatePoint(geom_way, pt_geom), tms::bigint
        FROM  (SELECT * FROM
        (WITH midtable AS (
        SELECT t.gid AS districtid,t.geom AS district_geom
        FROM %s AS t
        JOIN %s AS g
        ON ST_Intersects(g.pt_geom, t.geom)) 
        SELECT distinct(district_geom) from midtable)  AS districttb
        JOIN %s AS cn
        ON ST_Intersects(districttb.district_geom, cn.geom_way)) AS sttable, (%s) AS pts
        WHERE ST_DWithin(pt_geom, geom_way, 60) OR (%s ST_DWithin(pt_geom, geom_way, 50))
        ORDER BY path, st_distance
        ''' % (disrticttbnm, tracetbnm, streettbnm, dpStr, fwayQueryTxt)
    pts = FetchtoTurple(sqlCandstreet)  # 获得每个点可能的edge结果并存入变量

    if pts is None or pts == []:  # 若当前Trip没有GPS点附近有符合的道路
        ptsDf, nids = None, []
        print('No streets found within the tolerance of the trace.')
        print('You might want to check the projection of the streets table and trace, or the gpsError configuration parameter.')
        return
    ptsDf = pd.DataFrame(pts, columns=['nid', 'edge', 'dist', 'frcalong', 'secs']).set_index(
        'edge')  # 将可能的GPS点及其道路段集合放入DataFrame中

    # Step2:为上述GPS点对应dege表,追加对应edge信息,并追加和下一个匹配上GPS点之间距离
    nidStr = str(tuple(ptsDf.nid.unique())) if len(ptsDf.nid.unique()) > 1 else '(' + str(
        ptsDf.nid.iloc[0]) + ')'  # 寻找存在符合条件道路的GPS点序号
    cmd = '''SELECT gid AS path, ST_Distance(pt_geom,lag(pt_geom) OVER (order by gid))/1000 AS seglength_km
                        FROM (%s) AS pts
                        WHERE gid IN %s;''' % (dpStr, nidStr)
    traceSegs = FetchtoTurple(cmd)

    ptsDf = ptsDf.join(edgesDf).reset_index().set_index('nid')  # 为GPS点可能的道路,按照edge号进行
    ptsDf = ptsDf.join(
        pd.DataFrame(traceSegs, columns=['nid', 'seglength']).set_index('nid')).sort_index()  # 添加匹配点和后续点的距离
    if not len(np.unique(ptsDf.index)) == ptsDf.index.max() + 1:  # 将表格的GPS标记index,nid列进行恢复连续性
        lookup = dict(zip(np.unique(ptsDf.index), np.arange(ptsDf.index.max() + 1)))
        ptsDf.reset_index(inplace=True)
        ptsDf['nid'] = ptsDf.nid.map(lookup)
        ptsDf.set_index('nid', inplace=True)

    # Step3:按照距离符合的分布计算其似然值
    ptsDf['distprob'] = ptsDf.dist.apply(lambda x: distanceLL(x))
    nids = ptsDf.index.unique()  # 按照匹配上的GPS点数量,从0开始往后排
    ptsDf.loc[nids.max(), 'distprob'] *= weight_1stlast  # first point is dealt with in viterbi
    ptsDf['rownum'] = list(range(len(ptsDf)))

    PTDisTime = time.time() - starttime

    return (pts, ptsDf, traceSegs, nids, PTDisTime)


# 第二步,针对GPS点可能匹配的edge段的source和target而言,计算距离
def updateCostMatrix(edgesDf, ptsDf, costMatrix, distMatrix, streettbnm, maxNodes):
    starttime = time.time()  # 记录填充的开始时间
    nodeList = np.unique(
        edgesDf.loc[ptsDf.edge.unique(), ['source', 'target']].values.flatten())  # 获得途径edge所涉及到的全部可能source和target
    allNodesToDo = [n1 for n1 in nodeList if not all(
        [(n1, n2) in costMatrix for n2 in nodeList])]  # 只有在某node到其他node的Dijkstra距离都有了,该node才不进入该list
    nodesToDoList = [allNodesToDo[x:x + maxNodes] for x in
                     range(0, len(allNodesToDo), maxNodes)]  # 将allNodesToDo按照内存大小分成批,进行处理

    # 像costMatrix和distMatrix中追加给定source和target间的距离
    for nodesToDo_src in nodesToDoList:
        for nodesToDo_tgt in nodesToDoList:
            if nodesToDo_src and nodesToDo_tgt:
                cmd = '''SELECT start_vid, end_vid, sum(pgr.cost) AS pgrcost, sum(s.km) AS length_km
                      FROM %s s,
                           pgr_dijkstra('SELECT id, source, target, cost, reverse_cost FROM %s',
                                         ARRAY%s, ARRAY%s, True) AS pgr
                      WHERE s.id=pgr.edge
                      GROUP BY start_vid,end_vid;''' % (
                streettbnm, streettbnm, str(list(nodesToDo_src)), str(list(nodesToDo_tgt)))
                result = FetchtoTurple(cmd)
                costMatrix.update(
                    {((ff[0], ff[1]), ff[2]) if ff[2] >= 0 else ((ff[0], ff[1]), 10000000) for ff in result})
                distMatrix.update(
                    {((ff[0], ff[1]), ff[3]) if ff[3] >= 0 else ((ff[0], ff[1]), 10000000) for ff in result})

                # add route to/from same node
                costMatrix.update({((nn, nn), 0.0) for nn in nodesToDo_src})
                distMatrix.update({((nn, nn), 0.0) for nn in nodesToDo_src})

                # add route where pgr_dijkstra does not return a result, usually because of islands
                # 注,这里就是论文说的,把Dijkstra通不到的改为很大的数,即1万公里
                problemNodes = {((n1, n2), 10000000) for n1 in nodesToDo_src for n2 in nodesToDo_tgt if
                                (n1, n2) not in costMatrix}
                costMatrix.update(problemNodes)
                distMatrix.update(problemNodes)

    endtime = time.time()  # 记录结束时间
    FillDisMatrixTime = endtime - starttime
    return (costMatrix, distMatrix, FillDisMatrixTime)


# 第四步,利用Vitebi算法求解最佳edge路线
def viterbi(N, nids, weight_1stlast, max_skip, topologicalScores, skip_penalty):
    """Derived from https://phvu.net/2013/12/06/sweet-implementation-of-viterbi-in-python/
       Key changes are to work in log space, make use of self.matrices, and use sparse representations to save memory"""
    # obs=None simply steps through the K state probabilities in order, i.e. K = self.obsProb.shape[1]
    # if obs is None: obs=range(self.obsProb.shape[1])

    # to distinguish true zeros from empty cells, we will add 1 to all values here
    backpt = sparse.lil_matrix((N, nids.max() + 1),
                               dtype=np.int32)  # 稀疏矩阵,N行,nids.max()+1列：生成的data用来保存非0值,rows用来记录这些非0值索引

    # initialization
    # trellis_0为Series,值为第一个GPS点(可能有几个可能edge,都是有两行)的定位距离对应对数概率 * weight_1stlast
    trellis_0 = np.repeat(ptsDf.loc[[0], 'distprob'], 2) * weight_1stlast
    lastidx1, lastidx2 = 0, trellis_0.shape[0]
    for nid in nids[1:]:
        nToSkip = 0 if nid == len(
            nids) - 1 else max_skip  # Don't allow last point to be dropped, so set other probabilities very small

        idx1 = int(ptsDf.loc[max(0, nid - nToSkip), 'rownum'].min() * 2)
        idx2 = int(ptsDf.loc[nid, 'rownum'].max() * 2 + 2)
        iterN = idx2 - idx1  # npoints on this iteration

        # calculate the probabilities from the scores matrices
        transScoreArr = temporalScores[lastidx1:lastidx2, idx1:idx2].toarray()
        transScoreArr[transScoreArr == 0] = 1e10  # these are not true zeros, but empty cells.主要是将速度比值为0的改成十分小,防止出问题
        distratioArr = topologicalScores[lastidx1:lastidx2, idx1:idx2].toarray()
        distratioArr[distratioArr == 0] = 1e10  # these are not true zeros, but empty cells.主要是将速度比值为0的改成十分小,防止出问题

        LL = temporalLL(transScoreArr) + topologicalLL(distratioArr)

        # 1st term is the probability from the previous iteration. 2nd term is observation probability. 3rd term is the transition probability
        trellis_1 = np.nanmax(np.broadcast_to(trellis_0[:, None], (trellis_0.shape[0], iterN)) +
                              (np.repeat(ptsDf.loc[nid - nToSkip:nid, 'distprob'], 2) - skip_penalty * np.repeat(
                                  np.array(nid - ptsDf.loc[nid - nToSkip:nid].index) ** 2, 2))[:, None].T +
                              LL, 0)

        backpt[idx1:idx2, nid] = np.nanargmax(np.broadcast_to(trellis_0[:, None], (trellis_0.shape[0], iterN)) + LL, 0)[
                                 :, None] + lastidx1 + 1

        trellis_0 = trellis_1
        lastidx1, lastidx2 = idx1, idx2

    # termination
    tokens = [np.nanargmax(trellis_1) + idx1]
    for i in reversed(nids[1:]):
        tokens.append(backpt[tokens[-1], i] - 1)
    return tokens[::-1]


# 第五步,对于已匹配的Edge,针对中间断开或Uturn存在现象,修复其途径edge及Uturn影响
def fillRouteGaps(streettbnm, edgesDf, allowFinalUturn, route, uturns):
    # 需要输入的参数edgesDf,allowFinalUturn
    starttime = time.time()  # 开始时间
    """Takes the top route in the list (must be sorted!) and fills in gaps"""
    # remove duplicates
    keepPts = [0] + [ii + 1 for ii, rr in enumerate(route[1:]) if route[ii] != rr]  # 出现edge变化的第一个点(中间还在一个edge的就删去了)
    edgeList = [rr[0] for ii, rr in enumerate(route) if ii in keepPts]  # keepPts中点对应的edge
    # 以下:rr为(edge,和底图相比的方向),ii为序号(和keepPts中相同)
    nodeList = [int(edgesDf.loc[rr[0], 'target']) if rr[1] == 0 else
                int(edgesDf.loc[rr[0], 'source'])
                for ii, rr in enumerate(route) if ii in keepPts]  # 将edgeList所述的边按照车辆行驶方向连接起来(协调source和target)
    uturnList = [rr for ii, rr in enumerate(uturns) if ii + 1 in keepPts]  # 显示是否有Uturn

    if len(edgeList) == 1:  # 都在一个edge上的话,没必要填充了,直接返回
        bestRoute = edgeList
        uturnFrcs = []
        endtime = time.time()  # 结束时间
        FillGapTime = endtime - starttime  # 填充途径edge用的时间
        return (bestRoute, uturnFrcs, FillGapTime)

    firstEdge = True
    fullroute = [edgeList[0]]  # first edge
    uturnFrcs = [-1]
    keepPts.append(len(nids))  # this is for indexing (below)
    oNode = nodeList[0]  # origin for next route
    for ii, (edge, node, uturn, ptId) in enumerate(zip(edgeList[1:], nodeList[1:], uturnList, keepPts[1:-1])):
        # node is far end of the edge we are going to. We want the other end, called dNode
        if uturn in [3, 4]:  # uturn on first edge
            fullroute.append(fullroute[-1])
            frcs = ptsDf.loc[keepPts[ii]:keepPts[ii + 1] - 1].loc[
                ptsDf.loc[keepPts[ii]:keepPts[ii + 1] - 1].edge == route[ptId - 1][0], 'frcalong']
            uturnFrcs = uturnFrcs[:-1] + [(frcs.min(), frcs.max()), -1]
            oNode = edgesDf.target[fullroute[-1]] if edgesDf.source[fullroute[-1]] == oNode else edgesDf.source[
                fullroute[-1]]
        if uturn in [2, 4]:  # uturn on last edge
            dNode = node
        else:
            dNode = edgesDf.target[edge] if edgesDf.source[edge] == node else edgesDf.source[edge]

        if oNode != dNode:  # missing edges
            cmd = '''SELECT array_agg(edge) AS edges FROM pgr_dijkstra(
                        'SELECT id, source, target, cost, reverse_cost FROM %s', %s, %s, True);
                  ''' % (streettbnm, str(oNode), str(dNode))
            result = FetchtoTurple(cmd)
            if result[0][0] is None:  # 不允许有拓扑网断裂或某个最佳edge为isolated状态存在
                print('Could not match trace. Network may be disconnected or have islands.')
                bestRoute = None
                uturnFrcs = None
                endtime = time.time()  # 结束时间
                FillGapTime = endtime - starttime  # 填充途径edge用的时间
                return (bestRoute, uturnFrcs, FillGapTime)
            fullroute += result[0][0][:-1]
            uturnFrcs += [-1] * (len(result[0][0]) - 1)
        if uturn in [2, 4]:
            fullroute.append(edge)
            frcs = ptsDf.loc[keepPts[ii + 1]:keepPts[ii + 2]].loc[
                ptsDf.loc[keepPts[ii + 1]:keepPts[ii + 2]].edge == route[ptId][0], 'frcalong']
            uturnFrcs.append((frcs.min(), frcs.max()))
        fullroute.append(edge)
        if uturn == 1:
            frcs = ptsDf.loc[keepPts[ii]:keepPts[ii + 2] - 1].loc[
                ptsDf.loc[keepPts[ii]:keepPts[ii + 2] - 1].edge == route[ptId][0], 'frcalong']
            if fullroute[-1] == fullroute[-2] and uturnFrcs[
                -1] == -1:  # uturn already made on previous edge, but not counted
                uturnFrcs = uturnFrcs[:-1] + [(frcs.min(), frcs.max()), -1]
            else:
                uturnFrcs.append((frcs.min(), frcs.max()))
        else:
            uturnFrcs.append(-1)

        oNode = node  # starting point for next edge

    # Check whether there is a U-turn on the final edge -  issue #12
    if allowFinalUturn:
        for nid in reversed(nids[1:]):  # find out where the final edge starts
            if route[nid] != route[nid - 1]:
                break
        frcsAlong = ptsDf[ptsDf.edge == route[-1][0]].loc[nid:, 'frcalong']
        threshold = 0.1  # how many km the furthest GPS ping has to be along, in order to add a uturn
        if ((route[-1][1] == 0 and frcsAlong.max() * edgesDf.loc[route[-1][0], 'km'] > threshold) or
                (route[-1][1] == 1 and (1 - frcsAlong.min()) * edgesDf.loc[route[-1][0], 'km'] > threshold)):
            fullroute.append(fullroute[-1])
            uturnFrcs[-1] = (frcsAlong.min(), frcsAlong.max())
            uturnFrcs.append(-1)

    endtime = time.time()  # 结束时间
    FillGapTime = endtime - starttime  # 填充途径edge用的时间

    assert len(uturnFrcs) == len(fullroute)
    bestRoute = fullroute
    uturnFrcs = uturnFrcs
    return (bestRoute, uturnFrcs, FillGapTime)


# 第六步,对于已经得到的bestRoute和uturnFrcs,删除bestRoute中抖动 edge(例如A-B-A,则删除B);
# 若第一个GPS点在edge上行驶了不到20%,则该edge无效并删除
# 若最后一个GPS点某edge上行驶了不到5m
def cleanupRoute(edgesDf, d):
    # 外部变量 edgesDf
    global bestRoute, uturnFrcs

    # 如果第一个或最后一个edge上的实际行驶距离小于5m,则删除这些edge(这几乎不会是真正在上面行驶过的证据)
    lastEdgeLength = edgesDf.loc[bestRoute[-1], 'km']  # 抽取最后一个edge的距离
    frc = ptsDf.frcalong[ptsDf.rownum == int(d[-1] / 2)].values[0]
    if d[-1] % 2 == 1: frc = 1 - frc  # reverse
    while (len(bestRoute) > 1 and
           (lastEdgeLength * frc < 0.005 or (allowFinalUturn is False and bestRoute[-1] == bestRoute[-2]))):
        bestRoute = bestRoute[:-1]
        uturnFrcs = uturnFrcs[:-1]
        lastEdgeLength, frc = 1, 1

    firstEdgeLength = edgesDf.loc[bestRoute[0], 'km']
    frc = ptsDf.frcalong[ptsDf.rownum == int(d[0] / 2)].values[0]
    if d[0] % 2 == 0: frc = 1 - frc  # reverse
    # also get rid of duplicate first edges. This happens because the 1st edge must include at least 20% of its length, but this is bypassed if there is a U-turn
    while len(bestRoute) > 1 and (firstEdgeLength * frc < 0.005 or bestRoute[0] == bestRoute[1]):
        bestRoute = bestRoute[1:]
        uturnFrcs = uturnFrcs[1:]
        firstEdgeLength, frc = 1, 1

    if len(bestRoute) > 1 and bestRoute[0] == bestRoute[1]:
        bestRoute = bestRoute[1:]
        uturnFrcs = uturnFrcs[1:]

    # get rid of triple edge sequences (often because of a GPS 'splodge'
    while len(bestRoute) > 2:
        for ii, id in enumerate(bestRoute[2:]):
            if [id] * 2 == bestRoute[ii:ii + 2] and len(bestRoute) > 2:
                bestRoute = bestRoute[:ii] + bestRoute[ii + 2:]
                uturnFrcs = uturnFrcs[:ii] + uturnFrcs[ii + 2:]
                break  # do another full loop
        break  # break when no more triple duplicates

# 第七步(可选),准确获取车辆具体通过的linestring(起点和终点的对应转为sublinestring)
def getMatchedLineString(traceGPS, bestRoute, edgesDf, uturnFrcs):
    # PP:若轨迹小于3个有效匹配GPS点,则该轨迹不可食用
    if bestRoute is None:  # 若bestRoute因为匹配问题(小于3个有效匹配GPS点|中途的edge修复连接不上),则matchedLineString为None
        # 注,在正常情况下,matchedLineString是个str,用None表示不正常情况
        matchedLineString = None
        return nmatchedLineString

    # Step1:从当前轨迹对应GPS坐标集合中抽取第一个和最后一个GPS点信息
    stpt_lon = traceGPS.loc[traceGPS.index[[0]], ['lon']].iloc[0, 0]
    stpt_lat = traceGPS.loc[traceGPS.index[[0]], ['lat']].iloc[0, 0]
    startEndPts_part1 = 'ST_Transform(ST_SetSRID(ST_MakePoint(' + str(np.round(stpt_lon, 6)) + ',' + str(
        np.round(stpt_lat, 6)) + '),4326),' + '3395' + ')'
    edpt_lon = traceGPS.loc[traceGPS.index[[len(traceGPS) - 1]], ['lon']].iloc[0, 0]
    edpt_lat = traceGPS.loc[traceGPS.index[[len(traceGPS) - 1]], ['lat']].iloc[0, 0]
    startEndPts_part2 = 'ST_Transform(ST_SetSRID(ST_MakePoint(' + str(np.round(edpt_lon, 6)) + ',' + str(
        np.round(edpt_lat, 6)) + '),4326),' + '3395' + ')'
    startEndPts = []
    startEndPts.append(startEndPts_part1)
    startEndPts.append(startEndPts_part2)

    # 获取信息字典,并赋值route变量
    cmdDict = {'streetGeomCol': 'geom_way', 'streetsTable': 'cn3395_2po_4pgr', 'streetIdCol': 'id'}
    streetGeomCol = 'geom_way'  ###### 这里有一个道路表格中地理序列名字问题
    route = bestRoute

    if len(route) == 1:  # 由于轨迹GPS点仅对应一个edge,则抽出轨迹第一个+最后一个GPS点在edge上的frac,后依据方向,裁切好subLineString
        cDict = dict(cmdDict, **{'edge': str(route[0]), 'traceId': traceId, 'startGeom': startEndPts[0],
                                 'endGeom': startEndPts[1],
                                 'fromExtra': fromExtra, 'whereExtra': whereExtra})
        linestr = '''SELECT CASE WHEN stfrac = endfrac THEN Null
                      WHEN stfrac<endfrac THEN ST_Line_SubString(%(streetGeomCol)s, stfrac, endfrac)
                      ELSE ST_Reverse(ST_Line_SubString(%(streetGeomCol)s, endfrac, stfrac)) END AS geom
                FROM (SELECT ST_Line_Locate_Point(%(streetGeomCol)s, %(startGeom)s) AS stfrac,
                                     ST_Line_Locate_Point(%(streetGeomCol)s, %(endGeom)s) AS endfrac
                               FROM %(streetsTable)s WHERE %(streetIdCol)s=%(edge)s) AS pts,
                    %(streetsTable)s WHERE %(streetIdCol)s=%(edge)s''' % cDict
        matchedLineString = linestr
        return matchedLineString

    prevNode = edgesDf.source[route[0]] if edgesDf.target[route[0]] in edgesDf.ix[route[1]][
        ['source', 'target']].tolist() else edgesDf.target[route[0]]
    # This doesn't work - ST_LineMerge only handles a certain number of lines, it seems
    # linestr = 'UPDATE %s SET matched_line%s = merged_line FROM\n(SELECT ST_LineMerge(ST_Collect(st_geom)) AS merged_line FROM (\n'
    # Instead, we build up the string through iterating over the edges in the route
    routeFrcs = None  # frc of edge to use if we have a u turn
    linestr = '''SELECT ST_RemoveRepeatedPoints(merged_line) AS geom FROM
                    (SELECT ST_LineFromMultiPoint(ST_Collect((nodes).geom)) AS merged_line FROM
                        (SELECT lorder, st_dumppoints(st_geom) AS nodes FROM (\n'''

    def addEdge(edgesDf, linestr, ii, edge, reverse, routeFrcs, lineindex, geomField, extra=None):
        # 需传入的参数edgesDf
        streetsTable = 'cn3395_2po_4pgr'  ###### 这里有一个道路表格名字问题
        streetIdCol = 'id'  ###### 这里有一个道路表格中index序列名字问题
        if extra == None: extra = ' WHERE'  # allows for trip_id to be added as a table
        if reverse:  # linestring needs to be reversed
            geomStr = 'ST_Reverse(%s)' % geomField if routeFrcs in [None, (
            0, 1)] else 'ST_Reverse(ST_LineSubstring(%s,%s,%s))' % (geomField, routeFrcs[0], routeFrcs[1])
            prevNode = edgesDf.source[edge]
        else:
            geomStr = '%s' % geomField if routeFrcs in [None, (0, 1)] else 'ST_LineSubstring(%s,%s,%s)' % (
            geomField, routeFrcs[0], routeFrcs[1])
            prevNode = edgesDf.target[edge]
        linestr += '\t\t\t\tSELECT %s AS lorder, %s AS st_geom FROM %s%s %s = %s UNION ALL\n' % (
        str(lineindex), geomStr, streetsTable, extra, streetIdCol, str(edge))
        return linestr, prevNode

    for ii, edge in enumerate(route):
        if prevNode not in edgesDf.ix[edge][['source', 'target']].tolist():
            # need to repeat the last edge - an out and back situation
            assert prevNode in edgesDf.ix[route[ii - 1]][['source', 'target']].tolist()
            matchedLineString = None
            return matchedLineString
            # linestr, prevNode = addEdge(linestr, ii-1, route[ii-1], not(reverse), ii+0.5, streetGeomCol)
            # stophere
        reverse = True if prevNode == edgesDf.target[edge] else False
        if ii == 0:  # first point - don't need whole edge
            if reverse:
                geomField = 'ST_LineSubString(%(streetGeomCol)s, 0, ST_LineLocatePoint(%(streetGeomCol)s, %(startGeom)s))' % dict(
                    cmdDict, **{'startGeom': startEndPts[0]})
            else:
                geomField = 'ST_LineSubString(%(streetGeomCol)s, ST_LineLocatePoint(%(streetGeomCol)s, %(startGeom)s), 1)' % dict(
                    cmdDict, **{'startGeom': startEndPts[0]})
            routeFrcs = None
            extra = ' WHERE '
        elif ii == len(route) - 1:  # last point
            if reverse:
                geomField = 'ST_LineSubString(%(streetGeomCol)s, ST_LineLocatePoint(%(streetGeomCol)s, %(endGeom)s), 1)' % dict(
                    cmdDict, **{'endGeom': startEndPts[1]})
            else:
                geomField = 'ST_LineSubString(%(streetGeomCol)s, 0, ST_LineLocatePoint(%(streetGeomCol)s, %(endGeom)s))' % dict(
                    cmdDict, **{'endGeom': startEndPts[1]})
            routeFrcs = None
            extra = ' WHERE '
        else:
            geomField, extra = streetGeomCol, None
            if edge == route[ii + 1] and edge != route[ii - 1] and (
                    ii + 2 == len(route) or edge != route[ii + 2]):  # first edge in uTurn, but not if we have a triple
                # uturnFrcs==-1 means an unexpected Uturn
                routeFrcs = None if uturnFrcs[ii] == -1 else (uturnFrcs[ii][0], 1) if reverse else (0, uturnFrcs[ii][1])
                if routeFrcs is None: print('Warning. Missing Uturn in trace %s' % traceId)
            elif edge == route[ii - 1] and (ii == 1 or edge != route[ii - 2]):  # second edge in uTurn
                routeFrcs = None if uturnFrcs[ii - 1] == -1 else (0, uturnFrcs[ii - 1][1]) if reverse else (
                uturnFrcs[ii - 1][0], 1)
                if routeFrcs is None: print('Warning. Missing Uturn in trace %s' % traceId)
            else:
                routeFrcs = None
        linestr, prevNode = addEdge(edgesDf, linestr, ii, edge, reverse, routeFrcs, ii + 1, geomField, extra)

    linestr = linestr[:-11] + ') AS e ORDER BY lorder) AS m) AS p'  # remove last UNION ALL
    matchedLineString = linestr
    return matchedLineString


def writeMatchToPostgres(tripTablenm, newGeom, matchedLineString, Carplate,
                         Tripnum):  # 结合getMatchedLineString函数将结果导入到表格中
    # 添加新增geometry列
    cmd = '''SELECT AddGeometryColumn('%s','%s',3395,'LineString',2);''' % (tripTablenm, newGeom)
    RunSQLQuery(cmd)

    # 需要创建一个trip表格,其中包含一列newGeomName用来接收处理好的轨迹(LineString),用trip途径的edge和收尾按frac切分的部分edge连成一条线
    TIDcol = 'tid'  # trip表格的index列
    Carplatecol = 'platenum'  # trip表格的车牌号列
    cmd = '''UPDATE %s SET %s = geom FROM (%s) q
                 WHERE %s = '%s' AND %s = %s;''' % (
    tripTablenm, newGeom, matchedLineString, Carplatecol, Carplate, TIDcol, Tripnum)
    RunSQLQuery(cmd)


# 第八步(可选):将trip途经的修复好的edge的序号写入到trip表格中
def UpdateEgdeinfo(tripTablenm, Edgecolnm, bestRoute, Carplate, Tripnum):
    TIDcol = 'tid'  # trip表格的index列
    Carplatecol = 'platenum'  # trip表格的车牌号列
    cmd = '''UPDATE %s SET %s = ARRAY%s''' % (tripTablenm, Edgecolnm, str(bestRoute))
    cmd += '''\n        WHERE %s = '%s' AND %s = %s;''' % (Carplatecol, Carplate, TIDcol, Tripnum)
    RunSQLQuery(cmd)


# 第九步:获得仅含最终候选edge的ptsDf表
def GetLocPoint(FNptsDf, FNtablenm, tracetbnm, streettbnm, route):  # 获得最终的结果表格
    FNptsDf = FNptsDf.reset_index()
    # 删减表格并加入数据库
    del FNptsDf['rownum']
    del FNptsDf['clazz']
    del FNptsDf['kmh']
    del FNptsDf['km']
    del FNptsDf['reverse_cost']
    del FNptsDf['cost']
    del FNptsDf['target']
    del FNptsDf['source']
    DataframetoPgDB(FNptsDf, FNtablenm)

    # 得到真实的坐标结果并输出
    getreallocstr = '''
    WITH midtable AS (
    SELECT org.*,edge,frcalong,dist,distprob,seglength FROM %s org RIGHT JOIN %s mtb ON org.gid = mtb.nid
    )
    SELECT midtable.*,source,target,
    CAST(ST_X(ST_AsText(ST_Transform(ST_LineInterpolatePoint(geom_way,frcalong),4326))) AS decimal(10,6)) AS reallon,
    CAST(ST_Y(ST_AsText(ST_Transform(ST_LineInterpolatePoint(geom_way,frcalong),4326))) AS decimal(10,6)) AS reallat
    FROM midtable LEFT JOIN %s on midtable.edge = cn203395_2po_4pgr.id ORDER BY gid
    ''' % (tracetbnm, FNtablenm, streettbnm)
    ReallocData = FetchSQLQuery(getreallocstr)
    del ReallocData['pt_geom']

    # 获得车辆实际行驶方向对应的source和target编号(车辆真正行驶的方向和底图规定的不一定相同,这在双向道尤其明显)
    keepPts = [0] + [ii + 1 for ii, rr in enumerate(route[1:]) if route[ii] != rr]  # 出现edge变化的第一个点(中间还在一个edge的就删去了)
    edgeList = [rr[0] for ii, rr in enumerate(route) if ii in keepPts]  # keepPts中点对应的edge
    # 以下:rr为(edge,和底图相比的方向),ii为序号(和keepPts中相同)
    nodeList = [int(edgesDf.loc[rr[0], 'target']) if rr[1] == 0 else
                int(edgesDf.loc[rr[0], 'source'])
                for ii, rr in enumerate(route) if ii in keepPts]  # 将edgeList所述的边按照车辆行驶方向连接起来(协调source和target)
    AnchorPts = keepPts
    AnchorPts.append(ReallocData['gid'].max())
    keepPts = keepPts[:-1]
    ReallocData['ptsource'] = 0
    ReallocData['pttarget'] = 0
    # 进行赋值
    Moddata = []
    for num in range(len(keepPts)):
        if num != len(keepPts) - 1:
            stindex = AnchorPts[num]  # 起点index
            edindex = AnchorPts[num + 1] - 1  # 终点index
        else:
            stindex = AnchorPts[num]  # 起点index
            edindex = AnchorPts[num + 1] + 1  # 终点index
        edgedata = ReallocData[(ReallocData['gid'] >= stindex) & (ReallocData['gid'] <= edindex)]
        edgedata['pttarget'] = nodeList[num]
        Moddata.append(edgedata)
    ReallocData = pd.concat(Moddata)
    ReallocData = ReallocData.reset_index(drop=True)
    # 填充source和是否与geom_way方向相同的指示列
    ReallocData['samedir'] = 1  # 1表示相同,0表示相反
    for num in range(len(ReallocData)):
        Target = ReallocData.loc[ReallocData.index[[num]], ['target']].iloc[0, 0]
        PTTarget = ReallocData.loc[ReallocData.index[[num]], ['pttarget']].iloc[0, 0]
        if Target == PTTarget:
            ReallocData.loc[ReallocData.index[[num]], ['ptsource']] = \
            ReallocData.loc[ReallocData.index[[num]], ['source']].iloc[0, 0]
            ReallocData.loc[ReallocData.index[[num]], ['samedir']] = 1  # 方向相反
        else:
            ReallocData.loc[ReallocData.index[[num]], ['ptsource']] = \
            ReallocData.loc[ReallocData.index[[num]], ['target']].iloc[0, 0]
            ReallocData.loc[ReallocData.index[[num]], ['samedir']] = 0  # 方向相反
    # 删除涉及的表格
    droptablestr = '''DROP TABLE %s;''' % FNtablenm
    RunSQLQuery(droptablestr)
    
    return ReallocData
        
##################################################### 最终运行函数 ##############################################
# 定义几个重要输出路径:GPS匹配结果,对应edge结果
Matchedroot = r'D:\Trajectory\MatchedResult\\' # 例,最终匹配结果输出地址
Viaedgeroot = r'D:\Trajectory\Viaedge\\' # 例,车辆真实途径道路段结果输出地址
# 性能设置区-根据计算机性能设置
maxNodes = 100  # 在计算costMatrix中可能node间距离时使用,性能越好该值越大

# 按照速度不超过120KMH进行筛选设置区
logFn = r'C:\Users\Administrator\Desktop\Trajectory\log.txt' # 例,程序运行记录文件
preprocesstbnm = 'pptable'  # 进行速度删除的表格名称
newtraceTable = 'newpptableten' # 进行速度删除用到的中间表格
DelOverSpeed = True # 是否使用速度过快进行数据预处理
FillAccTrace = False # 是否对车辆途径路径进行优化
FillEdgeID = False # 是否对车辆途径路径进行优化

# 匹配参数设置区-1
fwayQuery = 'clazz<13 OR kmh>=60'  # 针对高速的具体筛选准则
weight_1stlast = 6  # 由于第一个函数对应候选edge对匹配成功影响很大,这个为针对1st和最后点的对数概率成倍值. This helps to ensure the match is closer to and from the start and endpoint
max_skip = 4  # 这里是计算edge间距离和时间的从某个GPS点往后找几个GPS点的参数.性能好的时候可以更大一些
skip_penaltyorg = 3  # penalty for skipping a point is temporalLL(skip_penalty) #
temporal_weight = 1.7  # how more more the temporal likelihood is weighted relative to the distance likelihood score
topol_weight = 1.7  # how more more the topological likelihood is weighted relative to the distance likelihood score
allowFinalUturn = True  # if True, allow a U-turn on the final edge

# 设置后续分析中使用的表格名
disrticttbnm = 'town'  # 乡镇行政区(含街道)表格名称
streettbnm = 'cn203395_2po_4pgr'  # 使用的全国道路表格名称

# 中间表格名
tracetbnm = 'trace_table' # 将轨迹数据导入PostgreSQL数据库后的表格名称
FNtablenm = 'matchedtable'  # 含有最终匹配结果的数据库表格

# STEP1.后续需构建edge之间的距离判断、时间判断等,需要读取所有数据库edge数据;同时针对那些速度低于15公里但能走通的方向,降低其cost以便于后期概率计算
fwayCols = 'clazz'  # street表格汇总存放道路等级的列
fwayColList = [cc.strip() for cc in fwayCols.split(',')]  # 考虑到可能有多个描绘道路等级的列
# 抽取道路
cmd = '''SELECT id, source::integer, target::integer, cost::real,reverse_cost::real, km::real,
               kmh::real,%s FROM %s;''' % (fwayCols, streettbnm)
edgesDf = pd.DataFrame(FetchtoTurple(cmd), columns=['edge', 'source', 'target', 'cost', 'reverse_cost', 'km',
                                                    'kmh'] + fwayColList).set_index('edge')
# 针对非重点egde段,降低其cost以便于后续概率计算
mask = (edgesDf.cost < 1000000) & (edgesDf.kmh < 15)
edgesDf.loc[mask, 'cost'] = edgesDf.loc[mask, 'cost'] * (edgesDf.loc[mask, 'kmh'] / 15.)
mask = (edgesDf.reverse_cost < 1000000) & (edgesDf.kmh < 15)
edgesDf.loc[mask, 'reverse_cost'] = edgesDf.loc[mask, 'reverse_cost'] * (edgesDf.loc[mask, 'kmh'] / 15.)
# 正式生成道路网可能node间的距离离散矩阵
maxN = max(edgesDf.source.max(), edgesDf.target.max()) + 1
costMatrix = sparse.dok_matrix((maxN, maxN))
distMatrix = sparse.dok_matrix((maxN, maxN))
# 启动costMatrix和distMatrix
try:
    costMatrix.update({})
except NotImplementedError:  # see https://github.com/scipy/scipy/issues/8338
    costMatrix.update = costMatrix._update
    distMatrix.update = distMatrix._update

# 参数设置区2 - 主要为uturnCost计算所有Edge的cost和reverse_cost中位数,必须在edgesDf后,才放到了后面
uturnCost = None  # 若此处不为None,则为一个认为给定的掉头值(时间,数据集中的cost和uturncost都是小时,即时间)
if uturnCost is None:  # not defined in config file
    uturncost = (edgesDf.cost.median() + edgesDf.reverse_cost.median()) / 2.
else:
    uturncost = float(uturnCost)

# Srtep7(可选):将匹配好的GPS轨迹line写入postgresql库中,有点慢,这个就是作图用,其他没用__开启开关
if FillAccTrace:
    tripTablenm = 'tripinfo'
    newGeom = 'modtripline'
    # 这里需要把所有trip的文件拼成一个大的dataframe,然后传入PostgreSQL中(前一个程序中完成,其他就不用做了)
else:
    pass
# Step8(可选):将trip途经的修复好的edge的序号写入到trip表格中__开启开关
if FillEdgeID:
    Edgecolnm = 'edgeidlist'
    # 这里需要把所有trip的文件拼成一个大的dataframe,然后传入PostgreSQL中(前一个程序中完成,其他就不用做了)
else:
    pass

# 读取轨迹数据
GPSfiledir = r'csv文件表示的轨迹文件地址,其他文件时则改写读取规则并最终保存为DataFrame即可'
traceGPS = pd.read_csv(GPSfiledir, engine='python', encoding='utf_8_sig')

# 删除瞬时速度高于120KM/H的运动点,剩余点小于3个时,则应放弃匹配 —— 记录有效定位小于3个的点的车及其轨迹序号
if DelOverSpeed:
    try:
        traceGPS = DelSpdError(traceGPS, logFn, preprocesstbnm,tracenum,newtraceTable)
    except:
        print('车轨迹数据出现错误')
        sys.exits()
    if traceGPS is None:
        print('车轨迹数据质量全部很差而无法使用')
        sys.exits()
    else:
        pass
else:
    pass

### 记录原始序号列
Trace_index = traceGPS['gid']  # 设置序号以分析后填充会去
### 更新index列
traceGPS = SQLResetIDcolumn(traceGPS, 'gid')
### 抽取新序号并增添前后Index对应dict
NewTrace_index = traceGPS['gid']
IndexInfo = {'New': NewTrace_index.values, 'Old': Trace_index.values}
IndexInfo = pd.DataFrame(IndexInfo)

# 导入数据库,并完成地图匹配
DataframetoPgDB(traceGPS, tracetbnm)

### XY转点 —— 暂定二维点
sqlXYPointstr = '''ALTER TABLE %s ADD COLUMN pt_geom geometry(Point, 4326);
    UPDATE %s SET pt_geom = ST_SetSRID(ST_MakePOINT(lon,lat),4326);''' % (tracetbnm, tracetbnm)
RunSQLQuery(sqlXYPointstr)

### 设置SIRD为3395
sqlTransProj = '''
ALTER TABLE %s ALTER COLUMN pt_geom TYPE Geometry(Point, 3395) USING ST_Transform(pt_geom,3395)
''' % tracetbnm
RunSQLQuery(sqlTransProj)


### Step1:进行最符合edge距离及似然值计算步骤
DisCalresult = MapDisToEdge(fwayQuery, tracetbnm, disrticttbnm, streettbnm, weight_1stlast)
if DisCalresult is None:
    print('车辆轨迹定位由于定位点周围没有候选道路而无法匹配')
    sys.exits()
else:
    pts = DisCalresult[0]  # GPS点其同可能edge的ID号等信息,Turple
    ptsDf = DisCalresult[1]  # GPS点同其可能edge的ID号\信息\似然值\匹配上点的前后距离,DataFrame
    traceSegs = DisCalresult[2]  # 匹配上的GPS点间距离,Turple
    nids = DisCalresult[3]  # 按照匹配上的GPS点数量,从0开始往后排
    PTDisTime = DisCalresult[4]  # 该步骤时间
del DisCalresult  # 及时删除GPS点可能edge这一步的内存

# 附近有候选edge的GPS点需高于3个才能继续
if len(nids) < 3:
    print('车辆轨迹定位数据中存在候选道路边的不足3个,无法匹配')
    sys.exits()

# Step2:根据GPS有对应edge的道路信息,填充稀疏矩阵CostMatrix,即两点间的Dijkstra距离
try:
    MatrixResult = updateCostMatrix(edgesDf, ptsDf, costMatrix, distMatrix, streettbnm, maxNodes)
except:
    print('由于dijkstra查询过多或下载的地图中存在没有连接上的路径而无法完成转移概率计算,请检查电子地图数据或重新设置maxNodes值')
    sys.exits()
costMatrix = MatrixResult[0]
distMatrix = MatrixResult[1]
FillTableTime = MatrixResult[2]
del MatrixResult  # 及时删除填充道路间距离这一步的内存

# Step3:寻找GPS点在候选EDGE间的距离差异和时间差异
N = len(ptsDf) * 2  # 生成承接上述时间/距离差的稀疏正方形矩阵边大小
rowKeys, colKeys, scores = [], [], []  # scores are held here, before conversion to a sparse matrix

for nid1, rr1 in ptsDf.loc[:nids[-2]].iterrows():
    rr1 = rr1.to_dict()
    for dir1 in [0, -1]:
        idx1 = int(rr1['rownum'] * 2 - dir1)
        # fill diagonal
        rowKeys.append(idx1)
        colKeys.append(idx1)
        scores.append((1e-10, 1, -1))  # 为了避免卡机
        seglength, lastnid = 0., -1
        for nid2, rr2 in ptsDf.loc[nid1 + 1:nid1 + 1 + max_skip].drop_duplicates(
                'edge').iterrows():  # max_skip is the maximum number of rows to skip. We drop duplicates because if this edge was done at a previous nid, we can skip
            rr2 = rr2.to_dict()
            if nid2 != lastnid:  # update seglength if a new nid is being entered, and pass it to the scores functions
                seglength += rr2['seglength']
                lastnid = nid2
            for dir2 in [0, -1]:
                rowKeys.append(idx1)
                colKeys.append(int(rr2['rownum'] * 2 - dir2))
                if rr1['edge'] == rr2['edge'] and dir1 == dir2:  # rr1和rr2在相同edge,方向相同
                    scores.append(transProbSameEdge(rr1, rr2, dir1, seglength))
                elif rr1['edge'] == rr2['edge'] or rr1[['target', 'source'][dir1]] != rr2[
                    ['target', 'source'][dir2]]:  # rr1和rr2在相同edge却方向相反;rr1和rr2在不同edge
                    scores.append(transProb(costMatrix, distMatrix, uturncost, rr1, rr2, dir1, dir2, seglength))
                else:
                    scores.append((1e10, 1e10, -1))

temporalScores = sparse.coo_matrix(([ii[0] for ii in scores], (rowKeys, colKeys)), shape=(N, N),
                                   dtype=np.float32).tocsr()
topologicalScores = sparse.coo_matrix(([ii[1] for ii in scores], (rowKeys, colKeys)), shape=(N, N),
                                      dtype=np.float32).tocsr()
uturns = sparse.coo_matrix(([ii[2] for ii in scores], (rowKeys, colKeys)), shape=(N, N), dtype=np.int8).tocsr()

# Step4:利用Viterbi算法求解最可能的道路
VIstart_time = time.time()  # 记录Vetebi算法开始时间
skip_penalty = abs(temporalLL(skip_penaltyorg))  # 必要参数
d = viterbi(N, nids, weight_1stlast, max_skip, topologicalScores, skip_penalty)

# Step5:修复车辆edge间断开等问题,并考虑Utrun情况
# 获得车辆走过的edge序号
route = [(ptsDf.loc[ptsDf.rownum == int(dd / 2), 'edge'].values[0], -1 * dd % 2) for dd in
         d]  # tuple of (edge, direction)

# 插入步Step9:获得车辆很多候选candidate的ptsDf中的最终保留结果,导入PGSQL中并获得真实定位,获取最终结果
FNptsDf = []  # GPS点最终对应的edge
for dd in d:
    FNptsDf.append(ptsDf.loc[ptsDf.rownum == int(dd / 2)])
FNptsDf = pd.concat(FNptsDf)  # 这里有的nid可能有多个记录,这主要是因为存在与地图edge正反向的问题
FNptsDf = FNptsDf.drop_duplicates(subset=['secs', 'edge'], keep='first', inplace=False)
ReallocData = GetLocPoint(FNptsDf, FNtablenm, tracetbnm, streettbnm, route) # 重要输出
ReallocData = pd.merge(ReallocData,IndexInfo,left_on = 'gid',right_on = 'New')
del ReallocData['New']
del ReallocData['gid']
Collist = ['Old'] + list(ReallocData)  # 创建最终的columns信息
Collistnm = ['gid'] + list(ReallocData)  # 创建最终的columns信息
ReallocData = ReallocData[Collist] # 重排顺序
ReallocData.columns = Collistnm # 修正名称
del ReallocData['Old']

MTGPSdir = Matchedroot + 'Matchedtrace.csv'
ReallocData.to_csv(MTGPSdir,index=False,encoding='utf_8_sig')


# 得到车辆在途中GPS点是否有U-turn出现
uturns = [uturns[n1, n2] for n1, n2 in zip(d[:-1], d[1:])]
# 填补车辆经过的具体edge
FillGapresult = fillRouteGaps(streettbnm, edgesDf, allowFinalUturn, route, uturns)
bestRoute = FillGapresult[0]
uturnFrcs = FillGapresult[1]
FillGapTime = FillGapresult[2]
del FillGapresult  # 及时删除变量,释放内存
# 判断修复结果
if bestRoute is None or -1 in bestRoute:
    print('由于dijkstra查询过多或下载的地图中存在没有连接上的路径而无法完成丢失定位数据处道路推测,请检查电子地图数据或重新设置maxNodes值')
    sys.exits()

# Step5: 对刚得到的bestRoute和uturnFrcs,针对GPS点晃(A-B-A)或第一个和最后一个edge不准确删除,进行修改
cleanupRoute(edgesDf, d)
# 此时该轨迹已经正确匹配了,标记其状态 | 为了加快速度不报错,错误的都在之前处理掉了
matchStatus = 0
# 此时可以为下一步评价匹配效果进行建设数据集,结果为LL,这个也可以不要
pointsToDrop = [ii + 1 for ii, dd in enumerate(zip(d[1:], d[:-1])) if dd[0] == dd[1]]
distLLs = ptsDf[ptsDf.rownum.isin([int(dd / 2) for dd in d])].distprob.describe()[['mean', 'min']].tolist()
distLLs[0] = distLLs[0] / weight_1stlast
distLLs[-1] = distLLs[-1] / weight_1stlast
temporalLLarray = temporalLL([temporalScores[n1, n2] for n1, n2 in zip(d[:-1], d[1:])])
topologicalLLarray = topologicalLL([topologicalScores[n1, n2] for n1, n2 in zip(d[:-1], d[1:])])
LL = distLLs + [temporalLLarray.mean(), temporalLLarray.min(), topologicalLLarray.mean(),
                topologicalLLarray.min()]

# 能运行到这里的,都是至少3个GPS点匹配上,并且连接起来edge了
# Srtep7(可选):将匹配好的GPS轨迹line写入postgresql库中,有点慢,这个就是作图用,其他没用
if FillAccTrace:
    matchedLineString = getMatchedLineString(traceGPS, bestRoute, edgesDf, uturnFrcs)
    Carplate = Tripfile.loc[Tripfile.index[[tracenum]], ['platenum']].iloc[0, 0]
    Tripnum = Tripfile.loc[Tripfile.index[[tracenum]], ['tid']].iloc[0, 0]
    writeMatchToPostgres(tripTablenm, newGeom, matchedLineString, Carplate, Tripnum)  # 向PostgreSQL中更新
else:
    pass
# Step8(可选):将trip途经的修复好的edge的序号写入到trip表格中
if FillEdgeID:
    Carplate = Tripfile.loc[Tripfile.index[[tracenum]], ['platenum']].iloc[0, 0]
    Tripnum = Tripfile.loc[Tripfile.index[[tracenum]], ['tid']].iloc[0, 0]
    UpdateEgdeinfo(tripTablenm, Edgecolnm, bestRoute, Carplate, Tripnum)  # 向PostgreSQL中更新
else:
    pass

# 构建最终的途径edge输出,platestr和tracenum已给出
sttimestr = ReallocData.loc[ReallocData.index[[0]], ['devtime']].iloc[0, 0]
edtimestr = ReallocData.loc[ReallocData.index[[len(ReallocData)-1]], ['devtime']].iloc[0, 0]
edgeinfostr = str(bestRoute)
ftdirstr = ReallocData.loc[ReallocData.index[[0]], ['samedir']].iloc[0, 0]
eddirstr = ReallocData.loc[ReallocData.index[[len(ReallocData)-1]], ['samedir']].iloc[0, 0]
ViaEdge = {'sttime':[sttimestr],'edtime':[edtimestr],'edgeinfo':[edgeinfostr],'fstdir':[ftdirstr],'lstdir':[eddirstr]}
ViaEdge = pd.DataFrame(ViaEdge)
VEDcollist = ['sttime','edtime','edgeinfo','fstdir','lstdir']
ViaEdge = ViaEdge[VEDcollist]
VEDdir = Viaedgeroot + 'Matchededge.csv'
ViaEdge.to_csv(VEDdir,index=False,encoding='utf_8_sig')