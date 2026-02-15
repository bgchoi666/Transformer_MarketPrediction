from fndata import Storage_module
import os

######### enterprise, ETF 구분 처리 아직 안 함########
if __name__=='__main__':
    #addr, host, pwd, schema = '@165.246.45.192', 'sunggil', 'ihq123!@#', 'financedb'
    addr, host, pwd, schema = '@165.246.196.138', 'vf-user', '!@value1419', 'financedb'
    conn = Storage_module(addr, host, pwd, schema)

    catagory_dir = './data/'
    category_list = os.listdir(catagory_dir) ## enterprise,ETF

    for catagory in category_list:
        if catagory == "enterprise" or catagory == "economy" or catagory == "market":
            subjects_dir = catagory_dir + catagory + '/' ## enterprise, economy, market
            subjects_flle_list = os.listdir(subjects_dir) ## enterprise : 종목별 9개 파일, ETF : 종목별 1개 파일
            subjects_flle_list.sort()

            conn.bad_file_list = list()

            for subject_file in subjects_flle_list:
                if subject_file != ".gitkeep":  # github 파일 제외
                    excel_path = subjects_dir + subject_file
                    conn.load_data(excel_path)
                    conn.upload_database()

            for bad_file in conn.bad_file_list:
                print(bad_file, "파일 점검대상.")

    print("파일로딩 종료!!!")
