# -*- coding: utf-8 -*-
import pymysql
from sqlalchemy import create_engine
import pandas as pd
pymysql.install_as_MySQLdb()
import numpy as np
import datetime

class Storage_module(object):
    def __init__(self, addr, host, pwd, schema):
        self.engine = create_engine("mysql+mysqldb://"+host+":"+pwd+addr+"/"+schema, encoding='utf-8',pool_recycle = 1800) #데이터베이스 엔진 생성

        self.df = None
        self.category = None
        self.item_code = None
        self.item_name = None
        self.subject = None
        self.table_name = None
        self.bad_file_list = list()

    def preprocessing_stock(self):
        self.df['락구분'] = np.where(self.df['락구분']=='정상', 1, self.df['락구분'])
        self.df['락구분'] = np.where(self.df['락구분']=='중간배당', 2, self.df['락구분'])
        self.df['락구분'] = np.where(self.df['락구분']=='액면분할', 3, self.df['락구분'])
        self.df['락구분'] = np.where(self.df['락구분']=='배당락', 4, self.df['락구분'])
        self.df['락구분'] = np.where(self.df['락구분']=='권리락', 5, self.df['락구분'])
        self.df['락구분'] = np.where(self.df['락구분']=='권배락', 6, self.df['락구분'])
        self.df['락구분'] = np.where(self.df['락구분']=='액면분할+권리락', 7, self.df['락구분'])
        self.df['락구분'] = np.where(self.df['락구분']=='액면병합', 8, self.df['락구분'])
        self.df['락구분'] = np.where(self.df['락구분']=='액면병합+권리락', 9, self.df['락구분'])
        self.df['락구분'] = np.where(self.df['락구분']=='감자', 10, self.df['락구분'])

        self.df['거래정지구분'] = np.where(self.df['거래정지구분']=='정상', 1, self.df['거래정지구분'])
        self.df['거래정지구분'] = np.where(self.df['거래정지구분']=='거래정지', 2, self.df['거래정지구분'])
        self.df['거래정지구분'] = np.where(self.df['거래정지구분']=='거래중단', 3, self.df['거래정지구분'])
        
        self.df['관리구분'] = np.where(self.df['관리구분']=='일반', 1, self.df['관리구분'])
        self.df['관리구분'] = np.where(self.df['관리구분']=='관리', 2, self.df['관리구분'])
        self.df['관리구분'] = np.where(self.df['관리구분']=='이상급등', 3, self.df['관리구분'])
        self.df['관리구분'] = np.where(self.df['관리구분']=='일반이상급등', 4, self.df['관리구분'])
        self.df['관리구분'] = np.where(self.df['관리구분']=='관리/일반이상급등', 5, self.df['관리구분'])
        self.df['관리구분'] = np.where(self.df['관리구분']=='우선주 급등감리', 6, self.df['관리구분'])
        self.df['관리구분'] = np.where(self.df['관리구분']=='관리/이상급등', 7, self.df['관리구분'])

        self.df['등락구분'] = np.where(self.df['등락구분']=='하한', 1, self.df['등락구분'])
        self.df['등락구분'] = np.where(self.df['등락구분']=='하락', 2, self.df['등락구분'])
        self.df['등락구분'] = np.where(self.df['등락구분']=='상승', 3, self.df['등락구분'])
        self.df['등락구분'] = np.where(self.df['등락구분']=='상한', 4, self.df['등락구분'])
        self.df['등락구분'] = np.where(self.df['등락구분']=='보합', 5, self.df['등락구분'])
        #self.df=self.df.dropna(axis = 1, how = 'all')

    def preprocessing_financial_ratio(self):
        self.df.replace('흑전', 99999, inplace=True) ## 흑자전환
        self.df.replace('적전', 0, inplace=True) ## 적자전환
        self.df.replace('적지', -99999, inplace=True) ## 적자지속
        self.df.replace('전기잠식', 88888, inplace=True) ## 잠식 : 기업의 자본총액이 자본금보다 낮아짐(적자인데 돈을 적자 메꾸는데 사용)
        self.df.replace('당기잠식', 11111, inplace=True) ## 전기 : 작년, 당기 : 올해, 완전잠식 : 자본금이 모두 사라짐
        self.df.replace('잠식지속', 22222, inplace=True)
        self.df.replace('완전잠식', -88888, inplace=True)
        self.df.replace('N/A(IFRS)', 0, inplace=True)

    def preprocessing_fixed_result_news(self):
        self.df['공시구분(확정실적속보)'] = np.where(self.df['공시구분(확정실적속보)']=='확정', 1, self.df['공시구분(확정실적속보)'])
        self.df['공시구분(확정실적속보)'] = np.where(self.df['공시구분(확정실적속보)']=='속보', 2, self.df['공시구분(확정실적속보)'])
        self.df['공시구분(확정실적속보)'] = np.where(self.df['공시구분(확정실적속보)']=='잠정', 3, self.df['공시구분(확정실적속보)'])

    def preprocessing_relative_value_indicator(self):
        self.df.replace('N/A(IFRS)', 0, inplace=True)
    
    def preprocessing_absolute_value_indicator(self):
        self.df.replace('N/A(IFRS)', 0, inplace=True)

    def load_data(self,xlsx_path):
        print("파일명", xlsx_path)

        self.category = xlsx_path.split('/')[-2]
        self.subject = '_'.join(xlsx_path.split('/')[-1].split('.')[0].split('_')[1:])  # 엑셀 파일 명으로 테이블 명 설정
        self.df = pd.read_excel(xlsx_path, sheet_name = 'Sheet1', header = 8) #전처리 되지 않은 excel 파일 load
        self.item_code = self.df.columns[1]  # 종목 item_code 파싱
        self.item_name = self.df.iloc[0,1] # 종목 명 파싱

        # 파일명과 종목명 비교 => 종목명 변경된 경우 또는 잘못 다운로드된 파일
        prefix = '_'.join(xlsx_path.split('/')[-1].split('.')[0].split('_')[0:1])
        if self.item_name != prefix:
            self.bad_file_list.append(xlsx_path)

        self.df.columns = list(self.df.iloc[3]) # 테이블 컬럼 설정
        self.df.columns = pd.io.parsers.ParserBase({'names':self.df.columns})._maybe_dedup_names(self.df.columns)
        self.df.columns.values[0] = 'date'
        self.df = self.df[5:] #시계열 데이터 추출

        col_size = self.df.shape[1]
        if self.subject == 'stock':
            self.preprocessing_stock()
        
        if self.subject == 'financial_ratio':
            self.preprocessing_financial_ratio()
            
        if self.subject == 'fixed_result_news':
            self.preprocessing_fixed_result_news()
            
        if self.subject == 'relative_value':
            self.preprocessing_relative_value_indicator()

        if self.subject == 'absolute_value_indicator':
            self.preprocessing_absolute_value_indicator()

        self.df.iloc[:, 1:col_size] = self.df.iloc[:, 1:col_size].astype(str).astype(float) #날짜를 제외한 모든 columns float 형변환
        #print(self.df.iloc[:, 1:col_size])
        print("data load complete!!!!!")

    def data_exist_check(self):
        table_check = self.engine.dialect.has_table(self.engine, self.table_name) # 기존 테이블 존재 여부 확인

        if table_check == True: #테이블이 존재하는 경우
            if self.category == 'enterprise' or self.category == 'index' or self.category == 'ETF':
                last_date = self.engine.execute(
                    "select date from " + self.table_name + " where item_code = '" + \
                        self.item_code + "' order by date desc limit 1").fetchall()  # 마지막 date검색
            else:
                last_date = self.engine.execute("select date from " + self.table_name + " order by date desc limit 1").fetchall()  # 마지막 date검색

            if not last_date: ## 데이터가 없을 경우
                return 'DN'
            
            else: ## 데이터가 있는 경우
                insert_date = last_date[0][0] + datetime.timedelta(days=1)
                # 기존에 DB에 존재하는 날의 다음 날
                self.df = self.df[self.df['date']>=insert_date]
                # Dataframe 자르기
                
                if self.df.empty: ## DB에 있는 데이터와 같을 경우
                    return 'SD'

                return 'DE'
                                  
        else : #테이블이 존재하지 않은 경우
            return 'TN'
        
    def set_item_code(self):
        self.df['item_code']=self.item_code  # 시계열 데이터에 item_code 컬럼 추가
        check = self.engine.execute ("select item_code from item where item_code = '"+self.item_code+"'").fetchall() # item 테이블에 종목 코드의 존재 여부 확인
        
        if not check: #item 테이블에 종목 코드가 존재 하지 않는 경우
            print('#Create stock item code')
            try:
                self.engine.execute("insert into item values('"+self.item_code+"','"+self.item_name+"','"+self.category+"')") #item 테이블에 종목 코드 추가
                print('#Successful item code addition')
            except Exception as ex :
                print('#Error occurred :',ex)
                
            
    def upload_database(self):
        self.table_name = self.category+'_'+self.subject
        data_status = self.data_exist_check() #테이블 조회하여 날짜 설정

        if data_status != 'SD':
            ##### DN : DB에 데이터 없음, SD : 같은 데이터 삽입, DE : DB에 데이터 존재, TN : 테이블 없음 
            if self.category == 'enterprise' or self.category == 'index' or self.category =='ETF':
                self.set_item_code()
                print(f"컬럼 수 : {len(self.df.columns)}")
                self.df.to_sql(name=self.table_name, con=self.engine,schema = "financedb", if_exists='append',index=False) # 데이터베이스로 데이터 업로드
                print(f"{self.item_name} {self.subject} #Upload successful")
            else:
                print(f"컬럼 수 : {len(self.df.columns)}")
                self.df.to_sql(name=self.table_name, con=self.engine, schema="financedb", if_exists='append', index=False)  # 데이터베이스로 데이터 업로드
                print(f"{self.item_name} {self.subject} #Upload successful")

        else:
             print(f"{self.item_name} {self.subject} #Same data insert")

        print("####################################################################################################")


class Extraction_module(object):
    def __init__(self, addr, host, pwd, schema):
        try:
            self.connection = pymysql.connect(host = addr ,port = 3306, user = host, password = pwd, db = schema, cursorclass = pymysql.cursors.DictCursor)
            self.cursor = self.connection.cursor()
            print(f'##Database connection successful!')
        except Exception as ex :
            print('#Error occurred :',ex)
        
        self.item_name = None
        self.item_info = None
        self.item_code = None
        self.full_items_info = None
        self.category_name = None
        self.dict_table_info = dict()
        self.cat_list = list()  
        self.item_table_list = list()
        self.dict_item_data = dict()
        self.dict_exg_data = dict()
        self.exg_table_list = list()

    def get_info_table(self):
        #######################전체 카테고리 정보 가져오기###########################
        query = 'select category_name from category;'
        self.cursor.execute(query)
        result = self.cursor.fetchall()

        self.cat_list = [i['category_name'] for i in result]
        '''
        for i in result:
            self.cat_list.append(i['category_name'])
        '''
        ######################전체 테이블 정보 가져오기############################
        
        for cat in self.cat_list:
            query = "show tables like '"+cat+"%'"
            self.cursor.execute(query)
            result = self.cursor.fetchall()

            table_list = [table['Tables_in_financedb ('+cat+'%)'] for table in result]
            '''
            table_list=list()
            for table in result:
                table_name = table['Tables_in_financedb ('+cat+'%)']
                table_list.append(table_name)
            '''
            self.dict_table_info[cat] = table_list
        
        return self.dict_table_info

    ##################################### Item Table에서 종목코드, 종목명, 카테고리명 추출##############
    def get_full_items_info(self):
        query = "select i.item_code, i.item_name, c.category_name as category from item i, category c where c.category_no = i.category;"
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        self.full_items_info = pd.DataFrame(result)
        return self.full_items_info 
        
    ######################특정 종목 정보 추출##################################
    def get_item(self,item_name):
        self.item_name = item_name
        self.item_info = self.full_items_info.iloc[np.where(self.full_items_info['item_name']== self.item_name)]
        self.item_code = list(self.item_info['item_code'])[0]
        self.category_name = list(self.item_info['category'])[0]
        self.item_table_list = self.dict_table_info[self.category_name]
        self.market_table_list = self.dict_table_info['market']
        self.economy_table_list = self.dict_table_info['economy']
        return self.item_info

    #####################Item data(enterprise, etf, index)table 추출#################
    def item_data_load(self, item_name = None, selected_table_list = None, merge = False):
        if item_name :
            self.get_info_table()
            self.get_full_items_info()
            self.get_item(item_name)

        if selected_table_list == None:
            selected_table_list = self.item_table_list

        for table_name in selected_table_list:
            query = "select *from "+table_name+" where item_code = (select item_code from item where item_name = '"+self.item_name+"');";
            self.cursor.execute(query)
            result = self.cursor.fetchall()

            if result != ():
            #print('테이블 결과' ,result)
                df = pd.DataFrame(result)
                #df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d').astype(str)
                #df.set_index(df['date'], inplace=True)
                self.dict_item_data[table_name] = df
                print(f'[success] {str(table_name)}')
        
        if merge == True :
            return self.merge_table_in_dict(self.dict_item_data)
        
        else :
            return self.dict_item_data

    ###################외생 변수 data(market, economy) table 추출###############################
    def exogenous_data_load(self, exg_category, selected_table_list=None, merge = False):
        self.dict_exg_data = dict()
        if len(self.dict_table_info.keys()) == 0:
            self.get_info_table()
        
        if selected_table_list == None:
            self.exg_table_list = self.dict_table_info[exg_category]
            selected_table_list = self.exg_table_list
        
        for table_name in selected_table_list :
            query = "select *from "+table_name+";";
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            
            df = pd.DataFrame(result)
            
            #df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d').astype(str)
            #df.set_index(df['date'], inplace=True)
            self.dict_exg_data[table_name] = df
            print(f'[success] {str(table_name)}')
        
        
        if merge == True :
            
            return self.merge_table_in_dict(self.dict_exg_data)
        
        else :
            return self.dict_exg_data
##########################data table 병합######################################
    def merge_table_in_dict(self,dict_table):
        #df = pd.concat(dict_table.values(), axis = 1)
        df = pd.concat(dict_table.values(), axis = 1)
        df.columns = pd.io.parsers.ParserBase({'names':df.columns})._maybe_dedup_names(df.columns) #중복컬럼명 처리
        return df


# 예측결과 DB 저장
class ResultUpload_module(object):
    def __init__(self, addr, host, pwd, schema):
        try:
            self.engine = create_engine("mysql+mysqldb://" + host + ":" + pwd + "@" + addr + "/" + schema, encoding='utf-8', pool_recycle=1800)
            print('##Database connection successful!')
        except Exception as ex:
            print('#Error occurred :', ex)

        self.param_id = None
        self.model = None
        self.time_step = None
        self.time_interval = None
        self.window_size = None
        self.prediction_day = None
        self.batch_size = None
        self.result_no = None

    # 예측파라메터(prediction_params) 조회(없으면 생성)
    def creation_prediction_params(self, model, time_step, time_interval, window_size, prediction_day, batch_size):
        self.model = model
        self.time_step = time_step
        self.time_interval = time_interval
        self.window_size = window_size
        self.prediction_day = prediction_day
        self.batch_size = batch_size

        query = "select param_id "+ \
                " from prediction_params where model='" + self.model + \
                "' and time_step='" + str(self.time_step) + \
                "' and time_interval='" + str(self.time_interval) + \
                "' and window_size='" + str(self.window_size) + "' and prediction_day='" + str(self.prediction_day) + \
                "' and batch_size='" + str(self.batch_size) + "' limit 1;"

        param_id = self.engine.execute(query).fetchall()

        if param_id != []:  ## 동일 파라메터가 있는 경우
            self.param_id = param_id[0]["param_id"]
            return param_id

        query = "select ifnull(max(param_id), 0) + 1 AS param_id " + " from prediction_params;"
        param_id = self.engine.execute(query).fetchall()
        self.param_id = param_id[0]["param_id"]
        try:
            query = "insert into " + "prediction_params values('" + str(self.param_id) + "',now(),'" + str(self.model) + "','" + str(self.time_step) + \
                    "','" + str(self.time_interval) + "','" + str(self.window_size) + "','" + str(self.prediction_day) + \
                    "','" + str(self.batch_size) + "');"  # 파라메터 추가
            self.engine.execute(query)
            print('#Successful prediction_params addition')
        except Exception as ex:
            print('#Error occurred :', ex)

        return self.param_id

    # 예측결과(prediction_result) 저장
    def creation_prediction_result(self, item_name, mape, mse, accuracy, pda, pram, pras, prcr30, prcr50, prcr70, time, detail):
        query = "select ifnull(max(result_no), 0) + 1 AS result_no " + " from prediction_result;"
        result_no = self.engine.execute(query).fetchall()
        self.result_no = result_no[0]["result_no"]

        try:
            # 예측결과 저장.
            query = "insert into " + "prediction_result (result_no, param_id, item_code, item_name, run_date, " + \
                    "mape, mse, accuracy, pasr, pamm, pamr, time, detail) " + \
                    "values('" + str(self.result_no) + "','" + str(self.param_id) + "'," + \
                    "(select item_code from item where item_name = '" + item_name + "')" + \
                    ",'" + item_name + "'," + "now()" + ",'" + str(mape) + "','" + str(mse) + "','" + str(accuracy) + \
                    "','" + str(pda) + "','" + str(pram) + "','" + str(pras) + "','" + str(prcr30) + "','" \
                    + str(prcr50) + "','" + str(prcr70) + "','"  + str(time) + "','" + detail + "');"  # 파라메터 추가
            self.engine.execute(query)
            print('#Successful prediction_result addition')
        except Exception as ex:
            print('#Error occurred :', ex)

    # 예측결과상세(prediction_result_detail) 저장
    def creation_prediction_result_detail(self, result_table):
        result_table["result_no"] = self.result_no
        result_table.columns = ["date", "ratio_real", "ratio_prediction", "price_real", "price_prediction", "result_no"]

        result_table.to_sql(name="prediction_result_detail", con=self.engine, schema="financedb", if_exists='append',
                            index=False)  # 데이터베이스로 데이터 업로드
        print('#Successful prediction_result_detail addition')

