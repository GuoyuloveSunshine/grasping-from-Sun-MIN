import os
import numpy as np
import json

def document_path_list(mvt_path):
    method_dir = ["blazepose","openpose", "kinect"]
    group_dir = os.listdir(mvt_path)
    dic_group = {}
    for group_name in group_dir:
        group_path = os.path.join(mvt_path,group_name)
#         print(group_path)
        dic_method = {}
        for method_name in method_dir:
            method_path = os.path.join(group_path,method_name)
            json_dir = os.listdir(method_path)
            
            file_read_list = []
            for json_name in json_dir:
                file_read_path = os.path.join(method_path,json_name)
                file_read_list.append(file_read_path)
#                 print(file_read_path)
            dic_method[method_name] = sorted(file_read_list)
        dic_group[group_name] = dic_method
    return dic_group

def kinToOpen_formule(kin_point,kin_ref,open_ref,dis_ref_kin,dis_ref_open):
    kin_point = np.array(kin_point)
    dis = np.linalg.norm(kin_point-kin_ref)
    dis_open = dis/dis_ref_kin*dis_ref_open
    return (kin_point-kin_ref)*dis_open+open_ref

## Convertir le coords de Kinect en coords de Openpose
def convertir_kinect_en_openpose(kin_temp,open_temp):
#     print(open_temp.keys())
    if 'lHip' not in open_temp.keys() or 'rHip' not in open_temp.keys():
        open_temp['mHip'] = [np.nan, np.nan]
    else:
        open_temp['mHip'] = [(open_temp['lHip'][0]+open_temp['rHip'][0])/2,(open_temp['lHip'][1]+open_temp['rHip'][1])/2]
    if 'mShoulder' not in open_temp.keys():
        open_temp['mShoulder'] = [np.nan, np.nan]
    dis_ref_open = np.linalg.norm([open_temp["mHip"][0]-open_temp["mShoulder"][0],open_temp["mHip"][1]-open_temp["mShoulder"][1]])
    dis_ref_kin = np.linalg.norm([kin_temp[0*7]-kin_temp[20*7],kin_temp[0*7+1]-kin_temp[20*7+1]])
    point_ref_open = np.array(open_temp["mShoulder"])
    point_ref_kin = np.array([kin_temp[20*7], kin_temp[20*7+1]]) #mShoulder
    kintoOpen = {}

    ## Body part
    kintoOpen["Head"]= np.hstack([kinToOpen_formule([kin_temp[3*7],kin_temp[3*7+1]],point_ref_kin,point_ref_open,dis_ref_kin,dis_ref_open),kin_temp[3*7+2]-kin_temp[20*7+2]])
    kintoOpen["mShoulder"] = np.hstack([kinToOpen_formule([kin_temp[20*7],kin_temp[20*7+1]],point_ref_kin,point_ref_open,dis_ref_kin,dis_ref_open),kin_temp[20*7+2]-kin_temp[20*7+2]])
    ## Right arm
    kintoOpen["rShoulder"] = np.hstack([kinToOpen_formule([kin_temp[8*7],kin_temp[8*7+1]],point_ref_kin,point_ref_open,dis_ref_kin,dis_ref_open),kin_temp[8*7+2]-kin_temp[20*7+2]])
    kintoOpen["rElbow"] = np.hstack([kinToOpen_formule([kin_temp[9*7],kin_temp[9*7+1]],point_ref_kin,point_ref_open,dis_ref_kin,dis_ref_open),kin_temp[9*7+2]-kin_temp[20*7+2]])
    kintoOpen["rWrist"] = np.hstack([kinToOpen_formule([kin_temp[10*7],kin_temp[10*7+1]],point_ref_kin,point_ref_open,dis_ref_kin,dis_ref_open),kin_temp[10*7+2]-kin_temp[20*7+2]])
    ## Left arm
    kintoOpen["lShoulder"] = np.hstack([kinToOpen_formule([kin_temp[4*7],kin_temp[4*7+1]],point_ref_kin,point_ref_open,dis_ref_kin,dis_ref_open),kin_temp[4*7+2]-kin_temp[20*7+2]])
    kintoOpen["lElbow"] = np.hstack([kinToOpen_formule([kin_temp[5*7],kin_temp[5*7+1]],point_ref_kin,point_ref_open,dis_ref_kin,dis_ref_open),kin_temp[5*7+2]-kin_temp[20*7+2]])
    kintoOpen["lWrist"] = np.hstack([kinToOpen_formule([kin_temp[6*7],kin_temp[6*7+1]],point_ref_kin,point_ref_open,dis_ref_kin,dis_ref_open),kin_temp[6*7+2]-kin_temp[20*7+2]])
    ## Right leg
    kintoOpen["rHip"] = np.hstack([kinToOpen_formule([kin_temp[16*7],kin_temp[16*7+1]],point_ref_kin,point_ref_open,dis_ref_kin,dis_ref_open),kin_temp[16*7+2]-kin_temp[20*7+2]])
    kintoOpen["rKnee"] = np.hstack([kinToOpen_formule([kin_temp[17*7],kin_temp[17*7+1]],point_ref_kin,point_ref_open,dis_ref_kin,dis_ref_open),kin_temp[17*7+2]-kin_temp[20*7+2]])
    kintoOpen["rAnkle"] = np.hstack([kinToOpen_formule([kin_temp[18*7],kin_temp[18*7+1]],point_ref_kin,point_ref_open,dis_ref_kin,dis_ref_open),kin_temp[18*7+2]-kin_temp[20*7+2]])
    ## Left leg
    kintoOpen["lHip"] = np.hstack([kinToOpen_formule([kin_temp[12*7],kin_temp[12*7+1]],point_ref_kin,point_ref_open,dis_ref_kin,dis_ref_open),kin_temp[12*7+2]-kin_temp[20*7+2]])      
    kintoOpen["lKnee"] = np.hstack([kinToOpen_formule([kin_temp[13*7],kin_temp[13*7+1]],point_ref_kin,point_ref_open,dis_ref_kin,dis_ref_open),kin_temp[13*7+2]-kin_temp[20*7+2]])
    kintoOpen["lAnkle"] = np.hstack([kinToOpen_formule([kin_temp[14*7],kin_temp[14*7+1]],point_ref_kin,point_ref_open,dis_ref_kin,dis_ref_open),kin_temp[14*7+2]-kin_temp[20*7+2]])
    return kintoOpen

def convertir_blazepose_en_openpose(bla_temp):
    sklt_bla = list(bla_temp.keys())
    deep_ref = (bla_temp[sklt_bla[11]][2]+bla_temp[sklt_bla[12]][2])/2
    blaztoOpen = {}
    blaztoOpen["Head"] = np.array(bla_temp[sklt_bla[0]])-np.array([0,0,deep_ref])
    blaztoOpen["mShoulder"] = np.array([(bla_temp[sklt_bla[11]][0]+bla_temp[sklt_bla[12]][0])/2,
                                        (bla_temp[sklt_bla[11]][1]+bla_temp[sklt_bla[12]][1])/2,
                                        (bla_temp[sklt_bla[11]][2]+bla_temp[sklt_bla[12]][2])/2])-np.array([0,0,deep_ref])
    ## Right arm
    blaztoOpen["rShoulder"] = np.array(bla_temp[sklt_bla[12]])-np.array([0,0,deep_ref])
    blaztoOpen["rElbow"] = np.array(bla_temp[sklt_bla[14]])-np.array([0,0,deep_ref])
    blaztoOpen["rWrist"] = np.array(bla_temp[sklt_bla[16]])-np.array([0,0,deep_ref])
    ## Left arm
    blaztoOpen["lShoulder"] = np.array(bla_temp[sklt_bla[11]])-np.array([0,0,deep_ref])
    blaztoOpen["lElbow"] = np.array(bla_temp[sklt_bla[13]])-np.array([0,0,deep_ref])
    blaztoOpen["lWrist"] = np.array(bla_temp[sklt_bla[15]])-np.array([0,0,deep_ref])
    ## Right leg
    blaztoOpen["rHip"] = np.array(bla_temp[sklt_bla[24]])-np.array([0,0,deep_ref])
    blaztoOpen["rKnee"] = np.array(bla_temp[sklt_bla[26]])-np.array([0,0,deep_ref])
    blaztoOpen["rAnkle"] = np.array(bla_temp[sklt_bla[28]])-np.array([0,0,deep_ref])
    ## Left leg
    blaztoOpen["lHip"] = np.array(bla_temp[sklt_bla[23]])-np.array([0,0,deep_ref])
    blaztoOpen["lKnee"] = np.array(bla_temp[sklt_bla[25]])-np.array([0,0,deep_ref])
    blaztoOpen["lAnkle"] = np.array(bla_temp[sklt_bla[27]])-np.array([0,0,deep_ref])

    return blaztoOpen  

def compare_blazepose_kinect_openpose_onevideo(blaze_file, kinect_file, open_file):
    print(blaze_file.name)
    print(kinect_file.name)
    print(open_file.name)
    print()
    open_dic = json.loads(open_file.read())
    open_all_frame = open_dic['positions']
    blaze_dic = json.loads(blaze_file.read())
    blaze_all_frame = blaze_dic['positions']
    kinect_all_frame = kinect_file.read().split("\n")
    frame_min = min(len(open_all_frame.keys()),len(blaze_all_frame.keys()),len(kinect_all_frame))
    list_sklt = ["Head","mShoulder",
                "rShoulder","rElbow","rWrist","lShoulder","lElbow","lWrist",
                "rHip","rKnee","rAnkle","lHip","lKnee","lAnkle"]
    Error_blazeOpen_frame = {}
    Error_KintOpen_frame = {}
    Error_blazeKint_frame_sansP = {}
    Error_blazeKint_frame_avecP = {}
    for sklt in list_sklt:
        Error_blazeOpen_frame[sklt] = np.zeros(frame_min)
        Error_KintOpen_frame[sklt] = np.zeros(frame_min)
        Error_blazeKint_frame_sansP[sklt] = np.zeros(frame_min)
        Error_blazeKint_frame_avecP[sklt] = np.zeros(frame_min)
#         print("k: ", len_kin,", b: ", len_bla)
    for frame in range(frame_min):
        blaze_temp = blaze_all_frame[str(frame+1)+".0"]
        kin_temp = np.array(kinect_all_frame[frame].split(" ")[:-1]).astype(float)
        open_sklt = open_all_frame[str(frame+1)+".0"]
#         print("frame: ", frame)
#         print(blaze_temp)
        blaze_sklt = convertir_blazepose_en_openpose(blaze_temp)
        if len(kin_temp) == 0:
            continue
        kin_sklt = convertir_kinect_en_openpose(kin_temp,open_sklt)
        for ske in list_sklt:
            if ske not in open_sklt.keys():
                Error_blazeOpen_frame[ske][frame] = np.nan
                Error_KintOpen_frame[ske][frame] = np.nan
                Error_blazeKint_frame_sansP[ske][frame] = np.nan
                Error_blazeKint_frame_avecP[ske][frame] = np.nan
            else:
                Error_blazeOpen_frame[ske][frame] = np.linalg.norm(blaze_sklt[ske][:-1] - np.array(open_sklt[ske]))
                Error_KintOpen_frame[ske][frame] = np.linalg.norm(kin_sklt[ske][:-1] - np.array(open_sklt[ske]))
                Error_blazeKint_frame_sansP[ske][frame] = np.linalg.norm(blaze_sklt[ske][:-1]-kin_sklt[ske][:-1])
                Error_blazeKint_frame_avecP[ske][frame] = np.linalg.norm(blaze_sklt[ske]-kin_sklt[ske])
    return Error_blazeOpen_frame, Error_KintOpen_frame, Error_blazeKint_frame_sansP, Error_blazeKint_frame_avecP

def get_error_dic(dic):
    Error_dic = {}
    list_sklt = ["Head","mShoulder",
                    "rShoulder","rElbow","rWrist","lShoulder","lElbow","lWrist",
                    "rHip","rKnee","rAnkle","lHip","lKnee","lAnkle"]
    list_error = ['Error_blazeOpen_par_mvt', 'Error_KintOpen_par_mvt', 
                    'Error_blazeKint_sansP_par_mvt','Error_blazeKint_avecP_par_mvt']

    for group in dic.keys():
        Error_group = {}
        blazepose_list = dic[group]['blazepose']
        openpose_list = dic[group]['openpose']
        kinect_list = dic[group]['kinect']
        lenlist = len(blazepose_list)
    #     print(len(blazepose_list))

    #     print(len(openpose_list))
    #     print(len(kinect_list))
        for Error_name in list_error:
            Error_group[Error_name] = {}
            for skl in list_sklt:
                Error_group[Error_name][skl] = {}
    #     print(Error_group)
        
        mvt_total = []
        for i in range(lenlist):
            print("%s, (%d/%d)"%(group, i+1, lenlist))
            path_name = blazepose_list[i]
            mvt = path_name.split('/')[-1].split('-')[-3]
            if mvt not in mvt_total:
                for Error_name in list_error:
                    for skl_name in list_sklt:
                        Error_group[Error_name][skl_name][mvt] = []
                mvt_total.append(mvt)
            
            open_file = open(openpose_list[i],'r+')
            blaze_file = open(blazepose_list[i],'r+')
            kin_file = open(kinect_list[i],"r+")
            Error_blazeOpen_one, Error_kinOpen_one, Error_blazeKint_sansP_one, Error_blazeKint_avecP_one = compare_blazepose_kinect_openpose_onevideo(blaze_file,kin_file,open_file)
            open_file.close()
            blaze_file.close()
            kin_file.close()
            for skl in list_sklt:
                Error_group["Error_blazeOpen_par_mvt"][skl][mvt].append(np.nanmean(Error_blazeOpen_one[skl]))
                Error_group["Error_KintOpen_par_mvt"][skl][mvt].append(np.nanmean(Error_kinOpen_one[skl]))
                Error_group["Error_blazeKint_sansP_par_mvt"][skl][mvt].append(np.nanmean(Error_blazeKint_sansP_one[skl]))
                Error_group["Error_blazeKint_avecP_par_mvt"][skl][mvt].append(np.nanmean(Error_blazeKint_avecP_one[skl]))
        Error_dic[group] = Error_group
    return Error_dic

if __name__ =="__main__":
    import pickle as pkl
    path_dic = document_path_list("../data")
    Error_dic = get_error_dic(path_dic)

    re = "../results"
    if not os.path.exists(re):
        os.mkdir(re)
    Error_path = os.path.join(re,"Error_all_group.pkl")

    with open(Error_path,'wb') as f:
        pkl.dump(Error_dic, f)