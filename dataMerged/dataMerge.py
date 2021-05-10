

def string_to_number(str):
  # if("." in str):
  #   try:
  #     res = float(str)
  #   except:
  #     res = str  
  # elif("-" in str):
  #   res = int(str)
  # elif(str.isdigit()):
  #   res = int(str)
  # else:
  #   res = str
  res = float(str)
  return(res)

def string_to_list(str):
  return [string_to_number(element) 
          for element in str.split()]

def input_file_zip(localpath):
    # desired joint value 
    f_q_des = open(localpath + "/q_sen.txt")
    f_dq_des = open(localpath + "/qdot_sen.txt")
    f_q_des.readline()
    f_dq_des.readline()

    f_q = open(localpath + "/q_sen.txt")
    f_dq = open(localpath + "/qdot_sen.txt")
    f_trq = open(localpath + "/trq.txt")

    f_base_ori = open(localpath + "/rot_base.txt")

    f_mag_al = open(localpath + "/magnetic_AL_foot_link.txt")
    f_mag_ar = open(localpath + "/magnetic_AR_foot_link.txt")
    f_mag_bl = open(localpath + "/magnetic_BL_foot_link.txt")
    f_mag_br = open(localpath + "/magnetic_BR_foot_link.txt")

    return zip(f_q, f_q_des, f_dq, f_dq_des, f_trq, f_base_ori, 
                f_mag_al, f_mag_ar, f_mag_bl, f_mag_br)

def output_file_list(localpath):
    f_q_des = open(localpath + "/q_des.txt", 'w')
    f_dq_des = open(localpath + "/qdot_des.txt", 'w')
    f_q = open(localpath + "/q_sen.txt", 'w')
    f_dq = open(localpath + "/qdot_sen.txt", 'w')
    f_trq = open(localpath + "/trq.txt", 'w')

    f_base_ori = open(localpath + "/rot_base.txt", 'w')

    f_mag_al = open(localpath + "/magnetic_AL_foot_link.txt", 'w')
    f_mag_ar = open(localpath + "/magnetic_AR_foot_link.txt", 'w')
    f_mag_bl = open(localpath + "/magnetic_BL_foot_link.txt", 'w')
    f_mag_br = open(localpath + "/magnetic_BR_foot_link.txt", 'w')

    return [f_q, f_q_des, f_dq, f_dq_des, f_trq, f_base_ori, 
                f_mag_al, f_mag_ar, f_mag_bl, f_mag_br]

def output_file_list_add(localpath):
    f_q_des = open(localpath + "/q_des.txt", 'a')
    f_dq_des = open(localpath + "/qdot_des.txt", 'a')
    f_q = open(localpath + "/q_sen.txt", 'a')
    f_dq = open(localpath + "/qdot_sen.txt", 'a')
    f_trq = open(localpath + "/trq.txt", 'a')

    f_base_ori = open(localpath + "/rot_base.txt", 'a')

    f_mag_al = open(localpath + "/magnetic_AL_foot_link.txt", 'a')
    f_mag_ar = open(localpath + "/magnetic_AR_foot_link.txt", 'a')
    f_mag_bl = open(localpath + "/magnetic_BL_foot_link.txt", 'a')
    f_mag_br = open(localpath + "/magnetic_BR_foot_link.txt", 'a')

    return [f_q, f_q_des, f_dq, f_dq_des, f_trq, f_base_ori, 
                f_mag_al, f_mag_ar, f_mag_bl, f_mag_br]


def writeoutput(f_in_zip, output_list, n_skip, n_num):

    # go to start position
    for i in range(n_skip):
        next(f_in_zip)

    niter = 0
    for input_lines in f_in_zip:
        # terminate
        if(niter > n_num):  return
        else:   niter = niter+1

        for line, fout in zip(input_lines,output_list):
            fout.writelines(line)



#######################################################################
#                       M A I N
#######################################################################
# q, q_des, dotq, dotq_des, trq, contact_al, f_mag_al, base_ori
global_path = '/home/jelee/GNN/graph-nets-physics/magneto-tf2-rotation-invariant'
output_folder = '/dataMerged/datafinal'

output_list = output_file_list(global_path + output_folder)

for i in range(12):
    input_folder = '/dataMerged/data' + str(i)    
    input_zip = input_file_zip( global_path + input_folder)

    chunk_size = 30

    writeoutput(input_zip, output_list, 600, chunk_size)
    writeoutput(input_zip, output_list, 200, chunk_size)
    writeoutput(input_zip, output_list, 200, chunk_size)
    writeoutput(input_zip, output_list, 200, chunk_size)
    writeoutput(input_zip, output_list, 200, chunk_size)
    writeoutput(input_zip, output_list, 200, chunk_size)
    writeoutput(input_zip, output_list, 200, chunk_size)
    writeoutput(input_zip, output_list, 200, chunk_size)
    writeoutput(input_zip, output_list, 200, chunk_size)
    writeoutput(input_zip, output_list, 200, chunk_size)
    writeoutput(input_zip, output_list, 200, chunk_size)
    writeoutput(input_zip, output_list, 200, chunk_size)
    writeoutput(input_zip, output_list, 200, chunk_size)
    writeoutput(input_zip, output_list, 200, chunk_size)
    writeoutput(input_zip, output_list, 200, chunk_size)
    writeoutput(input_zip, output_list, 200, chunk_size)
    writeoutput(input_zip, output_list, 200, chunk_size)

