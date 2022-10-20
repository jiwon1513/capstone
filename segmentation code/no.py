def movingAverage(lAngle, rAngle):
    ma_lAngle, ma_rangle = [], []
    Tm = 10

    for i in range(len(lAngle) - Tm):
        ma_lAngle.append(sum(lAngle[i : i + Tm]) / Tm)
        ma_rangle.append(sum(rAngle[i : i + Tm]) / Tm)

    return np.array(ma_lAngle), np.array(ma_rangle), Tm

elif groundTruth == "mhe":
if ma:
    ma_lpos, ma_rpos, Tm = movingAverage(lJPos[0], rJPos[0])
    lgtPoint = util.find_local_maximas(-ma_lpos, distance)
    rgtPoint = util.find_local_maximas(-ma_rpos, distance)

    offset = [Tm] * len(lgtPoint)
    lgtPoint = [a + b for a, b in zip(lgtPoint, offset)]
    rgtPoint = [a + b for a, b in zip(rgtPoint, offset)]
    new = lJPos[0][0:Tm].tolist() + ma_lpos.tolist()
    sp = SlidingPlot(2, data=[lJPos[0], new])
    sp.ylabel(["Hip Angle(deg)", "Hip Angle(deg)"])
    sp.legend([["LFEAngle"], ["LFEAngle"]])
else:
    lgtPoint = util.find_local_maximas(-lJPos[0], distance)
    rgtPoint = util.find_local_maximas(-rJPos[0], distance)
    sp = SlidingPlot(data=[lJPos[0]])
    sp.ylabel("Hip Angle(deg)")
    sp.legend(["LFEAngle"])

