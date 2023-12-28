#https://github.com/anujdutt9/Disease-Prediction-from-Symptoms/tree/master/saved_model
#pip3 install customtkinter
#https://github.com/Nasim992/Disease-Prediction-System/tree/master
import customtkinter as ct
from tkinter import messagebox as msg
from CTkMessagebox import CTkMessagebox as cmsg
#importing classes from train.py
import pandas as pd
#import pickle as pk
import joblib
import numpy as np

def login():
    def back(crpg) :  
        global page2,page1,page3 
        if crpg=="page2" : 
            #model=train_.skmodel()
            filename = 'joblib_model.sav'
            #joblib.dump(model, filename)
            cmsg(title="Success",message="Model created successfully.")
            page2.pack_forget()
            page1.pack_forget()
            mainwin()
        elif crpg=="page3" :
            page3.pack_forget()
            mainwin()
    def submit(symptoms) :
        global page3
        #global symptom,var,gen,symptoms
        #g = [float(x.get()) for x in symptoms.values()]
        for x in symptoms.keys():
            val=symptoms[x]

            try :
                symptoms[x]=int(val.get())
            except AttributeError:
                print("error avyo")
                pass
        symptoms.update({ 'altered_sensorium': 0,
                'red_spots_over_body': 0, 'belly_pain': 0, 'abnormal_menstruation': 0, 'dischromic _patches': 0, 'watering_from_eyes': 0,
                'increased_appetite': 0, 'polyuria': 0, 'family_history': 0, 'mucoid_sputum': 0, 'rusty_sputum': 0, 'lack_of_concentration': 0,
                'visual_disturbances': 0, 'receiving_blood_transfusion': 0, 'receiving_unsterile_injections': 0, 'coma': 0,
                'stomach_bleeding': 0, 'distention_of_abdomen': 0, 'history_of_alcohol_consumption': 0 ,'fluid_overload.1': 0,
                'blood_in_sputum': 0, 'prominent_veins_on_calf': 0, 'palpitations': 0, 'painful_walking': 0, 'pus_filled_pimples': 0,
                'blackheads': 0, 'scurring': 0, 'skin_peeling': 0, 'silver_like_dusting': 0, 'small_dents_in_nails': 0, 'inflammatory_nails': 0,
                'blister': 0, 'red_sore_around_nose': 0, 'yellow_crust_ooze': 0})
        print(symptoms)
        #s1=symptoms.values()
       
        #print(s1)
        #g.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0])
        # g=[]
        #rint(len(g),'= no of symptoms')
        # for x in s1:
        #     #print(x)
        #     g.append(float(x))
        '''dg=pd.DataFrame([g],columns=(['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain','altered_sensorium','red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
                'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration',
                'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
                'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption' ,'fluid_overload.1',
                'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples',
                'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
                'blister', 'red_sore_around_nose', 'yellow_crust_ooze']))
        #g=[[list(gen.get())[0],],[list(gen.get())[1],],[list(gen.get())[2],],[list(gen.get())[3],],[list(gen.get())[4],],[gen,]]'''
        df_test = pd.DataFrame(columns=list(symptoms.keys()))
        df_test.loc[0] = np.array(list(symptoms.values()))
        del symptoms
        disorder="some genetic disorder."
        #f=open('model.pkl','rb')
        filename = 'joblib_model.sav'
        model = joblib.load(str("./saved_model/decision_tree.joblib"))
        disorder=model.predict(df_test)
        print(df_test)
        print(disorder)
        cmsg(title="Disorder",message=f"You have {disorder}.")
        # page3.pack_forget()
        page3()
    def page3() :
        global gen,symptom,var,symptoms,page3,main
        main.geometry("920x475")
        frame.pack_forget()
        page3=ct.CTkFrame(main)
        page3.pack(padx=20,pady=60,fill="both",expand=True)
        page3._set_appearance_mode("dark")
        nmlabel=ct.CTkLabel(master=page3,text_color="lightsteelblue",text='Full Name: ',font=("Comfortaa",20))
        nmlabel.grid(row=0,column=0,padx=10,pady=20,sticky="w")
        nmentry=ct.CTkEntry(master=page3,width=300,height=40,placeholder_text="Name",font=("Comfortaa",20))
        nmentry.grid(row=0,column=1,padx=10,pady=20,columnspan=2)
        genlabel=ct.CTkLabel(master=page3,text_color="lightsteelblue",text='Gender:',font=("Comfortaa",20))
        #genlabel.grid(row=1,column=0,padx=10,pady=20,sticky="w")
        gen=ct.StringVar()
        rdM=ct.CTkRadioButton(page3,text="M",value='Male',variable=gen,font=("Comfortaa",15))
        rdM.grid(row=1,column=1,padx=10,pady=20)
        rdF=ct.CTkRadioButton(page3,text="F",value='Female',variable=gen,font=("Comfortaa",15))
        rdF.grid(row=1,column=2,padx=10,pady=20)
        rdO=ct.CTkRadioButton(page3,text="Other",value='Other',variable=gen,font=("Comfortaa",15))
        rdO.grid(row=1,column=3,padx=10,pady=20)
        symplabel=ct.CTkLabel(master=page3,text_color="lightsteelblue",text='Symptoms:',font=("Comfortaa",20))
        symplabel.grid(row=2,column=0,padx=10,pady=20,sticky="w")
        '''symptoms = {'itching': ct.IntVar(), 'skin_rash': ct.IntVar(), 'continuous_sneezing': ct.IntVar(),
                 'chills': ct.IntVar(), 'joint_pain': ct.IntVar(), 'stomach_pain': ct.IntVar(), 'acidity': ct.IntVar()
                , 'vomiting': ct.IntVar(), 'fatigue': ct.IntVar(),
                 'anxiety': ct.IntVar(), 'cold_hands_and_feets': ct.IntVar(),
                'restlessness': ct.IntVar(), 'cough': ct.IntVar(),
                'high_fever': ct.IntVar(), 
                'indigestion': ct.IntVar(), 'headache': ct.IntVar(), 
                'pain_behind_the_eyes': ct.IntVar(), 'constipation': ct.IntVar(),
                 'runny_nose': ct.IntVar(), 'congestion': ct.IntVar(), 'chest_pain': ct.IntVar(), 
                'knee_pain': ct.IntVar(), 'hip_joint_pain': ct.IntVar(), 'muscle_weakness': ct.IntVar(), 'stiff_neck': ct.IntVar(), 'swelling_joints': ct.IntVar(), 'movement_stiffness': ct.IntVar(),
                }'''
        # try:
        #     del symptoms
        # except:
        #     pass
        symptoms={'itching': ct.IntVar(), 'skin_rash': ct.IntVar(), 'nodal_skin_eruptions': ct.IntVar(), 'continuous_sneezing': ct.IntVar(),
                'shivering': ct.IntVar(), 'chills': ct.IntVar(), 'joint_pain': ct.IntVar(), 'stomach_pain': ct.IntVar(), 'acidity': ct.IntVar(), 'ulcers_on_tongue': ct.IntVar(),
                'muscle_wasting': ct.IntVar(), 'vomiting': ct.IntVar(), 'burning_micturition': ct.IntVar(), 'spotting_ urination': ct.IntVar(), 'fatigue': ct.IntVar(),
                'weight_gain': ct.IntVar(), 'anxiety': ct.IntVar(), 'cold_hands_and_feets': ct.IntVar(), 'mood_swings': ct.IntVar(), 'weight_loss': ct.IntVar(),
                'restlessness': ct.IntVar(), 'lethargy': ct.IntVar(), 'patches_in_throat': ct.IntVar(), 'irregular_sugar_level': ct.IntVar(), 'cough': ct.IntVar(),
                'high_fever': ct.IntVar(), 'sunken_eyes': ct.IntVar(), 'breathlessness': ct.IntVar(), 'sweating': ct.IntVar(), 'dehydration': ct.IntVar(),
                'indigestion': ct.IntVar(), 'headache': ct.IntVar(), 'yellowish_skin': ct.IntVar(), 'dark_urine': ct.IntVar(), 'nausea': ct.IntVar(), 'loss_of_appetite': ct.IntVar(),
                'pain_behind_the_eyes': ct.IntVar(), 'back_pain': ct.IntVar(), 'constipation': ct.IntVar(), 'abdominal_pain': ct.IntVar(), 'diarrhoea': ct.IntVar(), 'mild_fever': ct.IntVar(),
                'yellow_urine': ct.IntVar(), 'yellowing_of_eyes': ct.IntVar(), 'acute_liver_failure': ct.IntVar(), 'fluid_overload': ct.IntVar(), 'swelling_of_stomach': ct.IntVar(),
                'swelled_lymph_nodes': ct.IntVar(), 'malaise': ct.IntVar(), 'blurred_and_distorted_vision': ct.IntVar(), 'phlegm': ct.IntVar(), 'throat_irritation': ct.IntVar(),
                'redness_of_eyes': ct.IntVar(), 'sinus_pressure': ct.IntVar(), 'runny_nose': ct.IntVar(), 'congestion': ct.IntVar(), 'chest_pain': ct.IntVar(), 'weakness_in_limbs': ct.IntVar(),
                'fast_heart_rate': ct.IntVar(), 'pain_during_bowel_movements': ct.IntVar(), 'pain_in_anal_region': ct.IntVar(), 'bloody_stool': ct.IntVar(),
                'irritation_in_anus': ct.IntVar(), 'neck_pain': ct.IntVar(), 'dizziness': ct.IntVar(), 'cramps': ct.IntVar(), 'bruising': ct.IntVar(), 'obesity': ct.IntVar(), 'swollen_legs': ct.IntVar(),
                'swollen_blood_vessels': ct.IntVar(), 'puffy_face_and_eyes': ct.IntVar(), 'enlarged_thyroid': ct.IntVar(), 'brittle_nails': ct.IntVar(), 'swollen_extremeties': ct.IntVar()
                ,'excessive_hunger': ct.IntVar(), 'extra_marital_contacts': ct.IntVar(), 'drying_and_tingling_lips': ct.IntVar(), 'slurred_speech': ct.IntVar(),
                'knee_pain': ct.IntVar(), 'hip_joint_pain': ct.IntVar(), 'muscle_weakness': ct.IntVar(), 'stiff_neck': ct.IntVar(), 'swelling_joints': ct.IntVar(), 'movement_stiffness': ct.IntVar(),
                'spinning_movements': ct.IntVar(), 'loss_of_balance': ct.IntVar(), 'unsteadiness': ct.IntVar(), 'weakness_of_one_body_side': ct.IntVar(), 'loss_of_smell': ct.IntVar(),
                'bladder_discomfort': ct.IntVar(), 'foul_smell_of urine': ct.IntVar(), 'continuous_feel_of_urine': ct.IntVar(), 'passage_of_gases': ct.IntVar(), 'internal_itching': ct.IntVar(),
                'toxic_look_(typhos)': ct.IntVar(), 'depression': ct.IntVar(), 'irritability': ct.IntVar(), 'muscle_pain': ct.IntVar()}

        symptom_checkboxes = {}
        # ct.IntVar().set(10)
        var=ct.IntVar().get()
        i=0
        y=3
        for symptom, var in symptoms.items():
            
            
            symptom_checkboxes[symptom] = ct.CTkCheckBox(page3, text=symptom, variable=var,font=("Comfortaa",15))
            symptom_checkboxes[symptom].grid(row=y,column=i, padx=20, pady=5, sticky="w")
            i+=1
            if i%7==0:
                i=0
                y+=1

        gender=gen.get()
        sub=ct.CTkButton(master=page3,width=150,height=30,text='SUBMIT',font=("Comfortaa",15),hover_color="green",command=lambda: submit(symptoms))
        sub.grid(row=0,column=4,sticky="e",padx=10,pady=20)
        bck=ct.CTkButton(master=page3,width=150,height=30,text='<-BACK',font=("Comfortaa",15),border_color="red",hover_color="red",command=lambda: back("page3"))
        bck.grid(padx=10,pady=20,row=0,column=6)
    
    def page2() :
        global page1,page2,main
        main.geometry("517x535")
        page2=ct.CTkFrame(main)
        page2.pack(padx=20,pady=60,fill="both",expand=True)
        page2._set_appearance_mode("dark")
        page1.pack_forget()
        frame.pack_forget()
        grlabel=ct.CTkLabel(master=page2,text_color="lightsteelblue",text='Choose Graph',font=("Comfortaa",25))
        grlabel.grid(padx=10,pady=30,row=0,column=0,columnspan=3)
        # grb1=ct.CTkButton(master=page2,width=200,height=40,text='Gender-Symptom',font=("Comfortaa",15),command=lambda: data_.gensymptom(data_))
        # grb2=ct.CTkButton(master=page2,width=200,height=40,text='Gender-Disease',font=("Comfortaa",15),command=lambda: data_.gendisease(data_))
        # grb3=ct.CTkButton(master=page2,width=200,height=40,text='Disease-Symptom',font=("Comfortaa",15))
        # grb4=ct.CTkButton(master=page2,width=200,height=40,text='Gender Piechart',font=("Comfortaa",15),command=lambda: data_.genpie(data_))
        grb5=ct.CTkButton(master=page2,width=200,height=40,text='Age Line Graph',font=("Comfortaa",15))
        # grb1.grid(row=1,column=0,padx=20,pady=30)
        # grb2.grid(padx=20,pady=30,row=1,column=1)
        # grb3.grid(padx=20,pady=30,row=2,column=0)
        # grb4.grid(padx=20,pady=30,row=2,column=1)
        # grb5.grid(padx=20,pady=30,row=3,column=0)
        bck=ct.CTkButton(master=page2,width=200,height=40,text='<-BACK',font=("Comfortaa",15),border_color="red",hover_color="red",command=lambda: back("page2"))
        bck.grid(padx=20,pady=30,row=3,column=1)
    def page1() :
        global page1,frame
        page1=ct.CTkFrame(main)
        page1.pack(padx=20,pady=60,fill="both",expand=True)
        page1._set_appearance_mode("dark")
        frame.pack_forget()
        dslabel=ct.CTkLabel(master=page1,text_color="lightsteelblue",text='Enter the data size',font=("Comfortaa",25))
        dslabel.pack(padx=10,pady=30)
        dsentry=ct.CTkEntry(master=page1,width=200,height=40,placeholder_text="Maximum: 5000")
        dsentry.pack(padx=10,pady=30)
        # dsbutton=ct.CTkButton(master=page1,width=200,height=40,text='Submit',font=("Comfortaa",15),command=lambda: [page2(),data_.read(data_)])
        # dsbutton.pack(padx=10,pady=30)
        # data_.read(data_)
    def mainwin() :
        global main,frame
        main.geometry("871x475")
        #main.resizable(width=False,height=False)
        frame = ct.CTkFrame(master=main)
        frame.pack(padx=20,pady=60,expand=True,fill="both")
        label=ct.CTkLabel(master=frame,text_color="lightsteelblue",text='What would you like to do?',font=("Comfortaa",40))
        frame._set_appearance_mode("dark")
        label.pack(pady=40,padx=10)
        bt1=ct.CTkButton(master=frame,width=200,height=40,font=("Comfortaa",16),text="Create a Model",command=page1)
        bt1.pack(padx=20,pady=20)
        bt2=ct.CTkButton(master=frame,width=200,height=40,font=("Comfortaa",16),text="Use the Model",command=page3)
        bt2.pack(padx=20,pady=20)
        main.mainloop()    

        
    global frame,label,entry1,entry2,bt,main
    username,password = "username","password"
    #if entry1.get()=="u" and entry2.get()=="p" :
    msg.showinfo("Login","You logged in successfully.")
    win.destroy()
    main=ct.CTk()
    main.geometry("871x475")
    main.resizable(width=False,height=True)
    main._set_appearance_mode("dark")
    main.title("Genetic Disorder AI")
    mainwin()
    main.mainloop()
        
    # else : 
    #     msg.showinfo("Login","Username and password invalid.")
        
# win is the login window    
win=ct.CTk()
win.geometry("600x500")
win.title("Login")
win._set_appearance_mode("dark")
#locking the geometry of the window
win.resizable(width=False, height=False)

frame = ct.CTkFrame(master=win,width=300,height=300)
frame.pack(padx=20,pady=60,expand=True,fill="both")

label=ct.CTkLabel(master=frame,text_color="lightsteelblue",text='Login',font=("Comfortaa",45))
label.pack(pady=20,padx=10)

#username and password are asked through entries
entry1 = ct.CTkEntry(master=frame,width=200,height=40,placeholder_text='Username')
entry1.pack(pady=20,padx=10)

entry2 = ct.CTkEntry(master=frame,width=200,height=40,placeholder_text='Password',show='*')
entry2.pack(pady=20,padx=10)

#when submit button is pressed, login function is called
bt=ct.CTkButton(master=frame,width=200,height=40,text='Submit',font=("Comfortaa",15),command=login)
bt.pack(pady=20,padx=10)

win.mainloop()