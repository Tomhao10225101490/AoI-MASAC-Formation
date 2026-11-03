import json
import numpy as np
from scripts.render_masac_v4_run import calculate_clearance, choose_evaluation, get_status, mesh


def test_status_preserves_combined_causes_and_unknown():
    assert 'UNKNOWN' in get_status({})
    labels=get_status({'success':False,'static_collision':True,'boundary_collision':True,'truncated':True})
    for text in ('FAILURE','STATIC COLLISION','BOUNDARY','TIMEOUT'): assert text in labels
    assert 'TIMEOUT' not in get_status({'success':False})
    assert 'SUCCESS' in get_status({'success':True})


def test_select_evaluation_and_clearance():
    assert choose_evaluation([{'map_seed':2,'success':False},{'map_seed':3,'success':True}],3)['success'] is True
    assert choose_evaluation([{'map_seed':2},{'map_seed':3}],4)=={}
    pos=np.array([[[5.,5.,5.],[8.,5.,5.]]])
    boxes=np.array([[[0.,0.,0.],[2.,2.,2.]]])
    names,values=calculate_clearance(pos,boxes,None,None,[10,10,10],1.)
    assert names == ['static boundary','world boundary','other UAV']
    assert np.isclose(values[1][0],1.) and np.isclose(values[2][0],1.)
    json.dumps(mesh(boxes[0].astype(np.float32)))
