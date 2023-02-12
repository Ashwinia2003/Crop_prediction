import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split

# Load the CSV file into a pandas dataframe

st.set_page_config(page_title="Crop Prediction", page_icon=":seedling:", layout="wide",initial_sidebar_state="expanded")

def load_lottie(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

lottie1 = load_lottie("https://assets2.lottiefiles.com/packages/lf20_xd9ypluc.json")
lottie2 = load_lottie("https://assets8.lottiefiles.com/private_files/lf30_4lyswkde.json")


df = pd.read_csv('Crop_Prediction.csv')
df2 = pd.read_csv('DR.csv')

Columns=["Nitrogen","Phosphorus","Potassium","Temperature","Humidity","pH","Rainfall","Final_Crop"]
InputColumns=["Nitrogen","Phosphorus","Potassium","Temperature","Humidity","pH","Rainfall"]
MLType=["Logistic Regression","Random Forest","Decision Tree","Naive Bayes","Support Vector Classifier"]


converts_dict = {
    'Nitrogen': 'Nitrogen',
    'Phosphorus': 'Phosphorus',
    'Potassium': 'Potassium',
    'Temperature': 'Temperature',
    'Humidity': 'Humidity',
    'pH': 'pH',
    'Rainfall': 'Rainfall',
    'Final_Crop': 'Final_Crops'
}

states = ["ANDAMAN And NICOBAR ISLANDS","ARUNACHAL PRADESH","ASSAM","MEGHALAYA","MANIPUR",
"MIZORAM","NAGALAND","TRIPURA","WEST BENGAL","SIKKIM","ORISSA","JHARKHAND","BIHAR","UTTAR PRADESH",
"UTTARANCHAL","HARYANA","CHANDIGARH","DELHI","PUNJAB","HIMACHAL","JAMMU AND KASHMIR","RAJASTHAN","MADHYA PRADESH",
"GUJARAT","DADAR NAGAR HAVELI","DAMAN AND DUI","MAHARASHTRA","GOA","CHATISGARH","ANDHRA PRADESH","TAMIL NADU",
"PONDICHERRY","KARNATAKA","KERALA","LAKSHADWEEP"]

districts = {
    "ANDAMAN And NICOBAR ISLANDS": ["NICOBAR","SOUTH ANDAMAN","N & M ANDAMAN"],

    "ARUNACHAL PRADESH": ["LOHIT","EAST SIANG","SUBANSIRI F.D","TIRAP","ANJAW (LOHIT)","LOWER DIBANG","CHANGLANG","PAPUM PARE",
    "LOW SUBANSIRI","UPPER SIANG","WEST SIANG","DIBANG VALLEY","WEST KAMENG","EAST KAMENG","TAWANG","KURUNG KUMEY"],

    "ASSAM": ["CACHAR","DARRANG","GOALPARA","KAMRUP","LAKHIMPUR","NORTH CACHAR","NAGAON","SIVASAGAR","BARPETA","DHUBRI",
    "DIBRUGARH","JORHAT","KARIMGANJ","KOKRAJHAR","SHONITPUR","GOLAGHAT","TINSUKIA","HAILAKANDI","DHEMAJI(LAKHI","KARBI ANGLONG",
    "UDALGURI(DARA","KAMRUP METROP","CHIRANG(BONGAI","BAKSA BARPETA","BONGAIGAON","MORIGAON","NALBARI"],

    "MEGHALAYA": ["EAST KHASI HI","JAINTIA HILLS","EAST GARO HIL","RI-BHOI","SOUTH GARO HI","W KHASI HILL","WEST GARO HIL",],

    "MANIPUR": ["IMPHAL EAST","SENAPATI","TAMENGLONG","CHANDEL","UKHRUL","THOUBAL","BISHNUPUR","IMPHAL WEST","CHURACHANDPUR",],
    
    "MIZORAM": ["AIZAWL","CHAMPHAI","KOLASIB","LUNGLEI","CHHIMTUIPUI","LAWNGTLAI","MAMIT","SAIHA","SERCHHIP",],

    "NAGALAND": ["KOHIMA","TUENSANG","MOKOKCHUNG","DIMAPUR","WOKHA","MON",
    "ZUNHEBOTO","PHEK","KEPHRIE","LONGLENG","PEREN",],

    "TRIPURA": ["NORTH TRIPURA","SOUTH TRIPURA","WEST TRIPURA","DHALAI",],

    "WEST BENGAL": ["COOCH BEHAR","DARJEELING","JALPAIGURI","MALDA","SOUTH DINAJPUR","NORTH DINAJPUR","BANKURA","BIRBHUM",
    "BURDWAN","HOOGHLY","HOWRAH","PURULIA","MURSHIDABAD","NADIA","NORTH 24 PARG","SOUTH 24 PARG","EAST MIDNAPOR","WEST MIDNAPOR","KOLKATA",],

    "SIKKIM": ["NORTH SIKKIM","EAST SIKKIM","WEST SIKKIM","SOUTH SIKKIM",],

    "ORISSA": ["BALASORE","BOLANGIR","KANDHAMAL/PHU","CUTTACK","DHENKANAL","GANJAM","KALAHANDI","KEONDJHARGARH",
    "KORAPUT","MAYURBHANJ","PURI","SAMBALPUR","SUNDARGARH","BHADRAK","JAJPUR","KENDRAPARA","ANGUL","NAWAPARA",
    "MALKANGIRI","NAWARANGPUR","NAYAGARH","KHURDA","JHARSUGUDA","DEOGARH","RAYAGADA","GAJAPATI","JAGATSINGHAPU",
    "BOUDHGARH","SONEPUR",],

    "JHARKHAND": ["BOKARO","DHANBAD","DUMKA","HAZARIBAG","PALAMU","RANCHI","SAHIBGANJ","WEST SINGHBHUM","DEOGHAR","GIRIDIH",
    "GODDA","GUMLA","LOHARDAGA","CHATRA","KODERMA","EAST SINGHBHU","GARHWA","SERAIKELA-KHA","JAMTARA","LATEHAR","SIMDEGA",
    "KHUNTI(RANCHI","RAMGARH",],

    "UTTAR PRADESH": ["ALLAHABAD","AZAMGARH","BAHRAICH","BALLIA","BANDA","BARABANKI","BASTI","DEORIA","FAIZABAD","FARRUKHABAD",
    "FATEHPUR","GHAZIPUR","GONDA","GORAKHPUR","JAUNPUR","KANPUR NAGAR","KHERI LAKHIMP","LUCKNOW","MIRZAPUR","PRATAPGARH","RAE BARELI",
    "SITAPUR","SULTANPUR","UNNAO","VARANASI","SONBHADRA","MAHARAJGANJ","MAU","SIDDHARTH NGR","KUSHINAGAR","AMBEDKAR NAGAR","KANNAUJ",
    "BALRAMPUR","KAUSHAMBI","SAHUJI MAHARA","KANPUR DEHAT","CHANDAULI","SANT KABIR NGR","SANT RAVIDAS""SHRAVASTI NGR","AGRA","ALIGARH",
    "BAREILLY","BIJNOR","BADAUN","BULANDSHAHAR","ETAH","ETAWAH","HAMIRPUR","JALAUN""JHANSI","LALITPUR""MAINPURI","MATHURA","MEERUT",
    "MORADABAD","MUZAFFARNAGAR","PILIBHIT","RAMPUR","SAHARANPUR","SHAHJAHANPUR","GHAZIABAD","FIROZABAD","MAHOBA""MAHAMAYA NAGA",
    "AURAIYA","BAGPAT","JYOTIBA PHULE","GAUTAM BUDDHA","KANSHIRAM NAG",],

    "BIHAR": ["BHAGALPUR","EAST CHAMPARAN","DARBHANGA","GAYA","MUNGER","MUZAFFARPUR","WEST CHAMPARAN","PURNEA","GOPALGANJ",
    "MADHUBANI","AURANGABAD","BEGUSARAI","BHOJPUR","NALANDA","PATNA","KATIHAR","KHAGARIA","SARAN","MADHEPURA","NAWADA",
    "ROHTAS","SAMASTIPUR","SITAMARHI","SIWAN","VAISHALI","JAHANABAD","ARARIA","BANKA","BHABUA","JAMUI",
    "KISHANGANJ","SHEIKHPURA","SUPAUL","LAKHISARAI","SHEOHAR","ARWAL","SAHARSA",],

    "CHANDIGARH": ["CHANDIGARH",],

    "HARYANA": ["AMBALA","GURGAON","HISAR","JIND","KARNAL","MAHENDRAGARH","ROHTAK","BHIWANI",
    "FARIDABAD","KURUKSHETRA","SIRSA","SONEPAT(RTK)","YAMUNANAGAR","KAITHAL","PANIPAT","REWARI",
    "FATEHABAD","JHAJJAR","PANCHKULA","MEWAT","PALWAL(FRD)",],

    "UTTARANCHAL": ["ALMORA","CHAMOLI","DEHRADUN","GARHWAL PAURI","NAINITAL","PITHORAGARH",
    "GARHWAL TEHRI","UTTARKASHI","HARIDWAR","CHAMPAWAT","RUDRAPRAYAG","UDHAM SINGH N","BAGESHWAR",],

    "JAMMU AND KASHMIR": ["ANANTNAG","BARAMULLA","DODA","JAMMU","KATHUA","UDHAMPUR","BADGAM","KUPWARA",
    "PULWAMA","SRINAGAR","KARGIL","POONCH","BANDIPORE","GANDERWAL","KULGAM/(ANT)","SHOPAN","SAMBA","KISTWAR","REASI","RAMBAN(DDA)",],

    "HIMACHAL": ["BILASPUR","CHAMBA","KANGRA","KINNAUR","KULLU","MANDI","HAMIRPUR","SHIMLA","SIRMAUR","SOLAN","UNA",],

    "PUNJAB": ["AMRITSAR","BATHINDA","FEROZEPUR","GURDASPUR","HOSHIARPUR","JALANDHAR","KAPURTHALA","LUDHIANA","PATIALA","RUPNAGAR",
    "SANGRUR","FARIDKOT","MOGA","NAWANSHAHR","FATEHGARH SAH","MUKTSAR","MANSA","BARNALA","SAS NAGAR(MGA)","TARN TARAN",],

    "DELHI": ["NEW DELHI","CENTRAL DELHI","EAST DELHI","NORTH DELHI","NE DELHI","SW DELHI","NW DELHI","SOUTH DELHI","WEST DELHI",],

    "DADAR NAGAR HAVELI": ["DNH"],

    "GUJARAT": ["AHMEDABAD","BANASKANTHA","BARODA","BHARUCH","VALSAD","DANGS","KHEDA","MEHSANA","PANCHMAHALS",
    "SABARKANTHA","SURAT","GANDHINAGAR","NARMADA(BRC)","NAVSARI(VSD)","ANAND(KHR)","PATAN(MHSN)","DAHOD(PNML)","TAPI(SRT)",
    "AMRELI","BHAVNAGAR","JAMNAGAR","JUNAGADH","RAJKOT","SURENDRANAGAR","PORBANDAR",],

    "MADHYA PRADESH": ["BETUL","VIDISHA","BHIND","DATIA","DEWAS","DHAR","GUNA","GWALIOR","HOSHANGABAD","INDORE","JHABUA","MANDSAUR",
    "MORENA","KHANDWA","KHARGONE","RAISEN","RAJGARH","RATLAM","SEHORE","SHAJAPUR","SHIVPURI","UJJAIN","BHOPAL","HARDA","NEEMUCH","SHEOPUR",
    "BARWANI","ASHOKNAGAR(GNA","BURHANPUR","ALIRAJPUR(JBA)","BALAGHAT","CHHATARPUR","CHHINDWARA","JABALPUR","MANDLA","NARSINGHPUR","PANNA",
    "REWA","SAGAR","SATNA","SEONI","SIDHI","TIKAMGARH","KATNI","DINDORI","UMARIA","DAMOH","ANUPPUR(SHAHD","SINGRAULI",],

    "RAJASTHAN": ["BARMER","BIKANER","CHURU","SRI GANGANAGA","JAISALMER","JALORE","JODHPUR","NAGAUR","PALI","HANUMANGARH",
    "AJMER","ALWAR","BANSWARA","BHARATPUR","BHILWARA","BUNDI","CHITTORGARH","DUNGARPUR","JAIPUR","JHALAWAR",
    "JHUNJHUNU","KOTA","SAWAI MADHOPUR","SIKAR","TONK","UDAIPUR","DHOLPUR","BARAN","DAUSA","RAJSAMAND","KARAULI","PRATAPGARH(CHT",],

    "GOA": ["NORTH GOA","SOUTH GOA",],

    "TAMIL NADU": ["VELLORE","COIMBATORE","DHARMAPURI","KANYAKUMARI","CHENNAI","NILGIRIS","RAMANATHAPURA","SALEM","THANJAVUR","TIRUCHIRAPPAL",
    "TIRUNELVELI","ERODE","PUDUKKOTTAI","DINDIGUL","VIRUDHUNAGAR""SIVAGANGA","THOOTHUKUDI","TIRUVANNAMALA","NAGAPATTINAM","VILUPPURAM",
    "CUDDALORE","KANCHIPURAM","TIRUVALLUR","THENI","NAMAKKAL","KARUR","PERAMBALUR","TIRUVARUR""KRISHNAGIRI","ARIYALUR","TIRUPUR",],

    "CHATISGARH": ["BASTAR","BILASPUR""DURG","RAIGARH","RAIPUR","SURGUJA","RAJNANDGAON","DANTEWADA","KANKER (NORH","JANJGIR-CHAMP","KORBA",
    "JASHPUR","DHAMTARI","MAHASAMUND","KORIYA","KOWARDHA (KAB","NARAYANPUR","BIJAPUR",],

    "ANDHRA PRADESH": ["EAST GODAVARI","WEST GODAVARI","GUNTUR","KRISHNA","NELLORE","PRAKASAM","SRIKAKULAM","VISAKHAPATNAM","VIZIANAGARAM",
    "ADILABAD","HYDERABAD","KARIMNAGAR","KHAMMAM","MAHABUBNAGAR","MEDAK","NALGONDA","NIZAMABAD","WARANGAL","RANGAREDDY","ANANTAPUR",
    "CHITTOOR","KUDDAPAH","KURNOOL","VELLORE",],

    "MAHARASHTRA": ["RAIGAD","RATNAGIRI","THANE","SINDHUDURG","AHMEDNAGAR","DHULE","JALGAON","KOLHAPUR","NASHIK","PUNE","SANGLI","SATARA",
    "SOLAPUR","NANDURBAR","AURANGABAD","BEED","NANDED","OSMANABAD","PARBHANI","LATUR","JALNA","HINGOLI","AKOLA","AMRAVATI","BHANDARA",
    "BULDHANA","CHANDRAPUR","NAGPUR","YAVATMAL","WARDHA","GADCHIROLI","WASHIM","GONDIA",],

    "DAMAN AND DUI": ["DAMAN","DIU",],

    "LAKSHADWEEP": ["LAKSHADWEEP",],

    "KERALA": ["ALAPPUZHA","CANNUR","ERNAKULAM","KOTTAYAM","KOZHIKODE","MALAPPURAM","PALAKKAD",
    "KOLLAM","THRISSUR","THIRUVANANTHA","IDUKKI","KASARGOD","PATHANAMTHITTA","WAYANAD",],

    "KARNATAKA": ["UTTAR KANNADA","DAKSHIN KANDA","UDUPI","BELGAM","BIDAR","BIJAPUR","DHARWAD","GULBARGA","YADGIR","RAICHUR","BAGALKOTE","GADAG",
    "HAVERI","KOPPAL","BANGALORE RUR","BELLARY","CHIKMAGALUR","CHITRADURGA","KODAGU","HASSAN","KOLAR","MANDYA","MYSORE","SHIMOGA","TUMKUR",
    "BANGALORE URB","CHAMARAJANAGA","DAVANGERE","RAMNAGAR(BNGR)","CHICKBALLAPUR",],

    "PONDICHERRY": ["PONDICHERRY","KARAIKAL","MAHE","YANAM",],
   
}

tp = ["JANUARY","FEBRUARY","MARCH","APRIL","MAY","JUNE","JULY","AUGUST","SEPTEMBER","OCTOBER","NOVEMBER","DECEMBER",
"ANNUAL","Jan-Mar","Apr-Jun","Jul-Sep","Oct-Dec"]

def scatterPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("darkgrid")
    sns.scatterplot(data=df, x=x, y=y,hue="Final_Crop", size="Final_Crop", palette="deep", sizes=(20, 200), legend="full")
    plt.xlabel(converts_dict[x], fontsize=22)
    plt.ylabel(converts_dict[y], fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def barPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("darkgrid")
    sns.barplot(data=df, x=x, y=y)
    plt.xlabel(converts_dict[x], fontsize=22)
    plt.ylabel(converts_dict[y], fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def boxPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("darkgrid")
    sns.boxplot(x=x, y=y, data=df)
    sns.despine(offset=10, trim=True)
    plt.xlabel(converts_dict[x], fontsize=22)
    plt.ylabel(converts_dict[y], fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)


def predict_yield(humidityt, pHt, temperaturet, rainfallt, potassiumt, phosphorust, nitrogent):
    # This is a simple example. In reality, the calculation might be much more complex and might include machine learning algorithms.
    return (
        0.5 * humidityt * pHt * 0.3 * temperaturet
        + 0.5 * rainfallt + 0.3 * potassiumt + 0.3 * phosphorust + 0.3 * nitrogent
    )


def main():
    choice = option_menu(
            menu_title=None,
            options=["Home", "Data", "Analyzation", "Prediction"],
            icons=["house","bar-chart","activity","graph-up-arrow"],
            orientation="horizontal")


    if choice == "Home":
        with st.container():
            left_column, right_column = st.columns((3.5,1))
            with left_column:
                st.header("CROPS AND SOIL")
                st.write("<h1 style='font-size:18px'> Soil is a mixture of organic matter, minerals, gases, liquids, and countless microorganisms that together support life on Earth. \n" 
                "It is a complex and dynamic natural resource that is essential for growing crops and plants, supporting wildlife, and regulating the water cycle. \n"
                "Soil is formed from the weathering of rocks and minerals and the decomposition of organic matter. It is comprised of various layers, each with different physical and chemical properties. \n"
                "The top layer is called the surface soil, which is rich in organic matter and microorganisms. The subsoil is composed mainly of minerals, while the subsoil below that is composed of rock fragments. \n"
                "Soil also plays a crucial role in regulating the Earth's carbon cycle by storing carbon in the form of organic matter. Healthy soil is essential for sustainable agriculture and for maintaining the health of ecosystems and the planet as a whole.\n </h1>",unsafe_allow_html=True)
                st.write("")
            with right_column:
                st_lottie(lottie1, height=300, key="Plant-1")

        st.markdown("""---""")

        with st.container():
            left_column, right_column = st.columns((1,3.5))
            with left_column:
                st_lottie(lottie2,height=300,key="Plant-2")
            with right_column:
                st.write("")
                st.write("")
                st.subheader("Which Crops grow best in which soil?")
                st.write("<h1 style='font-size:18px'>The type of soil that a crop grows best in depends on several factors, including the crop's nutrient requirements, pH preference, and water-holding capacity. Here are some general guidelines for crops and the types of soil they prefer: \n"
                "Tomatoes, peppers, eggplants, and other vegetables in the nightshade family prefer well-drained soils with a slightly acidic pH of 6.0 to 6.5. Root crops like carrots, beets, and radishes do well in loose, well-drained soils with a pH range of 6.0 to 7.0. \n"
                "Potatoes grow best in well-drained, fertile soils with a pH range of 4.5 to 7.0. Broccoli, cauliflower, cabbage, and other cole crops prefer soils with a pH of 6.0 to 7.0 that are high in organic matter.\n"
                " Corn, beans, and other legumes are well-suited to fertile, well-drained soils with a pH. \n </h1>",unsafe_allow_html=True)

    if choice == "Data":
        with st.container():
            left_column,middle_column, right_column = st.columns((1,2,1))
            with middle_column:
                st.subheader("Overview of the Data :")
                dataframe=pd.read_csv("Crop_Prediction.csv")
                st.write(dataframe)
    
        with st.container():
            left_column,middle_column, right_column = st.columns((1,3,1))
            with middle_column:
                st.write("")
                st.write("The parameters that are used as inputs in this crop prediction model to estimate the expected growth of the crop. The model could use machine learning algorithms, statistical models, or other methods to analyze the relationships between these parameters and crop growth, and make predictions based on the input values.")
                st.markdown("""---""")
                st.subheader("Parameters used for this Prediction Model :")
                st.write(" ⁕ Humidity: High humidity levels can create a favorable environment for pests and diseases, while low humidity levels can cause water stress in plants. An optimal range of humidity for crop growth varies depending on the crop and the climate.")
                st.write(" ⁕ pH: The pH of the soil affects the availability of nutrients to the plants. Most crops grow best in soil with a pH between 6.0 and 7.0, but some crops have specific requirements. For example, blueberries prefer soil with a pH between 4.0 and 5.0.")
                st.write(" ⁕ Temperature: The temperature affects plant growth by affecting metabolic processes and water uptake. Each crop has a specific range of temperatures that is optimal for growth. For example, most crops grow best between 20°C and 30°C.")
                st.write(" ⁕ State: The state where the crop is grown can affect its growth due to regional differences in climate, soil type, and available water resources.")
                st.write(" ⁕ District: The district within the state can have further variations in climate, soil type, and water resources, affecting crop growth.")
                st.write(" ⁕ Potassium: Potassium is an essential nutrient for plant growth, and it plays a role in water regulation, photosynthesis, and disease resistance.")
                st.write(" ⁕ Phosphorus: Phosphorus is also an essential nutrient for plant growth, and it plays a role in energy storage and transfer, root growth, and flower and seed production.")
                st.write(" ⁕ Nitrogen: Nitrogen is a key nutrient for plant growth, and it plays a role in photosynthesis, chlorophyll synthesis, and protein production.")
                st.write(" ⁕ Rainfall: Adequate rainfall is important for crop growth, as it provides the water needed for photosynthesis and other metabolic processes. Too much or too little rainfall can both be problematic for crop growth.")

    if choice == "Analyzation":
        with st.container():
            left_column,middle_column, right_column = st.columns((2,0.2,2))
            with left_column:
                plot_type = st.selectbox("Select The Type Of Graph", ('Bar Graph', 'Scatter Plot', 'Box Plot'))
                st.subheader("Relation Between Soil Components")

                if plot_type == 'Bar Graph':
                    x = "Final_Crop"
                    y = st.selectbox("Select the Parameter to Plot",(["Nitrogen","Phosphorus","Potassium","Temperature","Humidity","pH","Rainfall"]))
                if plot_type == 'Scatter Plot':
                    x = st.selectbox("Select the Parameter-1",(["Nitrogen","Phosphorus","Potassium","Temperature","Humidity","pH","Rainfall"]))
                    y = st.selectbox("Select the Parameter-2",(["Nitrogen","Phosphorus","Potassium","Temperature","Humidity","pH","Rainfall"]))
                if plot_type == 'Box Plot':
                    x = "Final_Crop"
                    y = st.selectbox("Select the Parameter to Plot",(["Nitrogen","Phosphorus","Potassium","Temperature","Humidity","pH","Rainfall"]))
                
                if st.button("View The Graph"):
                    with right_column:
                        if plot_type == 'Bar Graph':
                            barPlotDrawer(x, y)
                        if plot_type == 'Scatter Plot':
                            scatterPlotDrawer(x, y)
                        if plot_type == 'Box Plot':
                            boxPlotDrawer(x, y)

    if choice == "Prediction":
        with st.container():
            left_column,middle_column, right_column = st.columns((1,3,1))
            with middle_column:
                st.header("Choose the Parameters to predict the right crop for the land")
                state = st.selectbox("Select a State", states)

# Show the list of districts for the selected state
                if state in districts:
                    district = st.selectbox("Select a district", districts[state])
                else:
                    st.write("No districts available for the selected state")
        
                month = st.selectbox("Select a Month or Time Period",tp)

                row1_value = state
                row2_value = district
                column_name = month
                result = df2.loc[(df2['STATE_UT_NAME'] == row1_value) & (df2['DISTRICT'] == row2_value), column_name].iloc[0]
                RainFallValue=result

                n = st.number_input('Enter the NITROGEN LEVEL', 0, 140)
                p = st.number_input('Enter the PHOSPHORUS LEVEL', 5, 145)
                k = st.number_input('Enter the POTASSIUM LEVEL', 5, 205)

                temperature = st.number_input('Enter the TEMPERATURE', 5, 45)
                humidity = st.number_input('Enter the HUMIDITY', 10, 100)
                ph = st.number_input('Enter the pH VALUE', 1, 10)
                rainfall = RainFallValue
            
                features_list=[n,p,k,temperature,humidity,ph,rainfall]

                MLmodel = st.selectbox("Choose a Machine Learning Model :",MLType)
            
                if st.button("Submit"):
                        df = pd.read_csv("Crop_Prediction.csv")
                        X = df[InputColumns]
                        y = df["Final_Crop"]

                        # Splitting the Train and Test Data
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 42)
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                        st.markdown("""---""")

                        if MLmodel=="Logistic Regression":
                            clf = LogisticRegression() 
                            clf.fit(X, y)
                            joblib.dump(clf, "clf.pkl")
                            clf = joblib.load("clf.pkl")
                            X_new = pd.DataFrame([features_list], columns = [InputColumns])
                            # Predicting the output based on the input parameters given
                            prediction = clf.predict(X_new)[0]
                            st.write("The Crop best suited for this region is :", prediction)
                            cv_scores = cross_val_score(clf, X, y, cv=5)[0]
                            st.write("Accuracy while using Logistic Regression :", cv_scores)
                            st.write("Accuracy Percentage :",cv_scores*100,"%")
    

                        if MLmodel=="Random Forest":
                            RF = RandomForestClassifier(n_estimators=20, random_state=0)
                            RF.fit(X,y)
                            RF = joblib.load("RandomForest.pkl")
                            data = pd.DataFrame([features_list], columns=InputColumns)
                            predicted_values = RF.predict(data)[0]
                            st.write("The Crop best suited for this region is :", predicted_values)
                            score = cross_val_score(RF,X,y,cv=5)[0]
                            st.write("Accuracy while using Random Forest :", score)
                            st.write("Accuracy Percentage :",score*100,"%")

                        if MLmodel=="Decision Tree":
                            DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)
                            DecisionTree.fit(X,y)
                            DT = joblib.load("DecisionTree.pkl")
                            data2 = pd.DataFrame([features_list], columns=InputColumns)
                            predicted_values2 = DecisionTree.predict(data2)[0]
                            st.write("The Crop best suited for this region is :", predicted_values2)
                            score = cross_val_score(DT,X,y,cv=5)[0]
                            st.write("Accuracy while using Decision Tree :", score)
                            st.write("Accuracy Percentage :",score*100,"%")

                        if MLmodel=="Naive Bayes":
                            NaiveBayes = GaussianNB()
                            NaiveBayes.fit(X,y)
                            NB = joblib.load("NBClassifier.pkl")
                            data3 = pd.DataFrame([features_list], columns=InputColumns)
                            predicted_values3 = NaiveBayes.predict(data3)[0]
                            st.write("The Crop best suited for this region is :", predicted_values3)
                            score = cross_val_score(NB,X,y,cv=5)[0]
                            st.write("Accuracy while using Naive Bayes :", score)
                            st.write("Accuracy Percentage :",score*100,"%")


                        if MLmodel=="Support Vector Classifier":
                            SVM = SVC(gamma='auto')
                            SVM.fit(X,y)
                            joblib.dump(SVM, "clf.pkl")
                            SVM = joblib.load("clf.pkl")
                            data4 = pd.DataFrame([features_list], columns = [InputColumns])
                            predicted_values4 = SVM.predict(data4)[0]
                            st.write("The Crop best suited for this region is :", predicted_values4)
                            score = cross_val_score(SVM,X,y,cv=5)[0]
                            st.write("Accuracy while using Support Vector Classifier :", score)
                            st.write("Accuracy Percentage :",score*100,"%")

if __name__=='__main__':
    main()