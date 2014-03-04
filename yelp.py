import pandas as pd
import json
from datetime import datetime as dt

def load_train_data():
    with open('./yelp_test_set/yelp_test_set_business.json') as f:
        data = [json.loads(line) for line in f]
        business = pd.DataFrame.from_dict(data)
    with open('./yelp_test_set/yelp_test_set_checkin.json') as f:
        data = [json.loads(line) for line in f]
        checkin = pd.DataFrame.from_dict(data)
    with open('./yelp_test_set/yelp_test_set_review.json') as f:
        data = [json.loads(line) for line in f]
        review = pd.DataFrame.from_dict(data)
    with open('./yelp_test_set/yelp_test_set_user.json') as f:
        data = [json.loads(line) for line in f]
        user = pd.DataFrame.from_dict(data)
    return business,checkin,review,user

def get_unique_items_from_arrays_in_series(series):
    toret = []
    for arr in series:
        for cat in arr:
            if cat not in toret:
                toret.append(cat)
    return toret

def get_unique_items_from_dicts_in_series(series):
    toret = []
    for dic in series:
        for cat in dic.keys():
            if cat not in toret:
                toret.append(cat)
    return toret

def days_from_date(date,ref=dt(2013,01,19)):
    ymd = date.split('-')
    year = int(ymd[0])
    month = int(ymd[1])
    day = int(ymd[2])
    return (ref - dt(year,month,day)).days
    
##X['review_age'] = X['date'].apply(try1.days_from_date)
##X['review_funny'] = X['votes'].map(lambda x: x['funny'])
##X['review_useful'] = X['votes'].map(lambda x: x['useful'])
##X['review_cool'] = X['votes'].map(lambda x: x['cool'])

##X['review_length'] = X['text'].map(lambda x:len(x.split()))

##X['total_checkins'] = X['checkin_info'].map(lambda x: sum(x.values()) if type(x) == dict else np.nan)

##review['caps'] = review['text'].map(lambda x: np.sum(np.array(x.split()) == np.array(x.upper().split()))-np.sum(np.array(x.split())=='I')-len(re.findall("[0-9]",x)))
##>>> review['caps'] = review['caps'].map(lambda x: x if x >= 0 else 0)
##review['$'] = review['text'].map(lambda x: len(re.findall("\$",x)))
##traindata = traindata.drop(['business_id','date','review_id','text','user_id','categories','city','full_address','business_name','neighborhoods','name','type','checkin_info'],axis=1)

##model.fit(traindata[traindata['review_useful']<20][['user_cool', 'user_funny', 'review_length', 'user_review_count', 'review_age', 'user_useful', 'review_stars', 'business_review_count','user_stars','caps','$']].dropna(),traindata[traindata['review_useful']<20][['user_cool', 'user_funny', 'review_length', 'user_review_count', 'review_age', 'user_useful', 'review_stars', 'business_review_count','user_stars','caps','$','review_useful']].dropna()['review_useful'])
##[user_cool, user_funny, review_length, user_review_count,
##review_age, user_useful, review_stars, business_review_count,
##longitude, user_stars, latitude, caps, $, checkins_friday, checkins_night,
##checkins_morning, checkins_afternoon, business_stars, checkins_evening,
##checkins_sunday, checkins_latenight, total_checkins, checkins_monday,
##checkins_earlymorning, checkins_thursday, checkins_tuesday,
##checkins_wednesday, open, state]

##model.fit(traindata[traindata['review_useful']<20][['review_length','review_age', 'review_stars', 'business_review_count','caps','$','latitude','longitude','business_stars','open']].dropna(axis=1),traindata[traindata['review_useful']<20][['review_length','review_age', 'review_stars', 'business_review_count','caps','$','review_useful','latitude','longitude','business_stars','open']].dropna(axis=1)['review_useful'])
##[review_length, review_age, longitude, latitude, business_review_count,
##review_stars, caps, $, business_stars, open, state]

##testdata1 = testdata.dropna()
##>>> testdata1['prediction'] = model.predict(testdata1[['user_cool', 'user_funny', 'review_length', 'user_review_count', 'review_age', 'user_useful', 'review_stars', 'business_review_count','user_stars','caps','$']])
##testdata2 = testdata.dropna(axis=1)
##testdata2['prediction'] = model.predict(testdata2[['review_length','review_age', 'review_stars', 'business_review_count','caps','$','latitude','longitude','business_stars','open']])
##predictions = testdata1[['review_id','prediction']].merge(testdata2[['review_id','prediction']],on='review_id',how='outer')

##trainuserstars = traindata.dropna(subset=['user_stars'])
##model.fit(trainuserstars[['review_stars', 'review_age', 'review_length', 'longitude', 'business_stars', 'latitude', 'business_review_count', '$', 'caps']].dropna(axis=0),trainuserstars[['review_stars', 'review_age', 'review_length', 'longitude', 'business_stars', 'latitude', 'business_review_count', '$', 'caps', 'user_stars']].dropna(axis=0)['user_stars'])
##model.fit(trainuserstars[['review_stars', 'review_age', 'review_length']].dropna(axis=0),trainuserstars[['review_stars', 'review_age', 'review_length', 'user_funny']].dropna(axis=0)['user_funny'])
##
##>>> traindata['user_stars']=traindata['user_stars'].fillna(value=9999,inplace=True)
##>>> for i in range(len(traindata)):
##	if traindata.ix[i]['user_stars']==9999:
##		traindata['user_stars'][i]=model.predict(traindata.ix[i][['review_stars', 'review_age', 'review_length', 'longitude', 'business_stars', 'latitude', 'business_review_count', '$', 'caps']])[0]
## 70 trees <5000 for user_funny,useful and cool <2000 for user_review_count
## 70 trees user_stars all data

def split_time(i,time):
	toret = 0
	for j in i.keys():
		hr = int(j.split('-')[0])
		if time == 'morning':
			if hr >= 8 and hr <= 11:
				toret += i[j]
		elif time == 'afternoon':
			if hr >= 12 and hr <= 15:
				toret += i[j]
		elif time == 'evening':
			if hr >= 16 and hr <=19:
				toret +=i[j]
		elif time == 'night':
			if hr >= 20 and hr <=23:
				toret +=i[j]
		elif time == 'latenight':
			if hr >= 0 and hr <=3:
				toret +=i[j]
		elif time == 'earlymorning':
			if hr >= 4 and hr <=7:
				toret +=i[j]
	return toret

def split_day(i,day):
	toret = 0
	for j in i.keys():
		d = int(j.split('-')[1])
		if day == d:
			toret += i[j]
	return toret

##X['checkins_sunday'] = X['checkin_info'].map(lambda x: split_day(x,0) if type(x) == dict else np.nan)
##days = ['sunday','monday','tuesday','wednesday','thursday','friday']
##for day in days:
##	X['checkins_'+day] = X['checkin_info'].map(lambda x: split_day(x,days.index(day)) if type(x) == dict else np.nan)

##>>> d = dict()
##>>> d1 = dict()
##>>> d2 = dict()
##>>> len(X[X['review_useful']==0])
##95370
##>>> for i in X[X['review_useful']==0]['categories']:
##	for j in i:
##		if j in d.keys():
##			d[j] += 1
##		else:
##			d[j] = 1
##
##			
##>>> len(X[X['review_useful']>2])
##37770
##>>> for i in X[X['review_useful']>2]['categories']:
##	for j in i:
##		if j in d1.keys():
##			d1[j] += 1
##		else:
##			d1[j] = 1
##
##			
##>>> for i in d.keys():
##	if i in d1.keys():
##		d2[i] = (d[i]*37770)/(d1[i]*95370)
##	else:
##		d2[i] = 9999
##>>> for w in sorted(d2,key=d2.get,reverse=True):
##      if d2[w] < 9999 and d2[w] > 1:
##	    print w, d2[w]

##########################################
##MORE GOOD REVIEWS THAN BAD - Categories
##########################################
##Adult Entertainment 11
##African 9
##Rafting/Kayaking 8
##Public Transportation 7
##Cooking Schools 6
##Animal Shelters 5
##Diagnostic Imaging 5
##Psychics & Astrologers 5
##Gift Shops 5
##Laser Eye Surgery/Lasik 4
##Basque 4
##Community Service/Non-Profit 3
##Mountain Biking 3
##Russian 3
##Boat Charters 3
##Festivals 3
##Television Stations 3
##Comedy Clubs 3
##Transportation 3
##Adult 3
##Outlet Stores 3
##Burmese 3
##Live/Raw Food 3
##Champagne Bars 3
##Local Flavor 3
##Plus Size Fashion 3
##Architects 2
##Hiking 2
##Travel Services 2
##Lounges 2
##Scandinavian 2
##Watch Repair 2
##Landscape Architects 2
##Costumes 2
##Climbing 2
##Mortgage Brokers 2
##Lakes 2
##Tapas Bars 2
##Swimwear 2
##Art Schools 2
##Jazz & Blues 2
##Cheese Shops 2
##Personal Chefs 2
##Dance Clubs 2
##Radio Stations 2
##Farmers Market 2
##Gay Bars 2
##Botanical Gardens 2
##Filipino 2
##Accountants 2
##Barre Classes 2
##Skating Rinks 2
##Leather Goods 2
##Thrift Stores 2
##Polish 2
##Kitchen & Bath 2
##Horse Racing 2
##Modern European 2
##Airlines 2
##Spanish 2

############################################
##MORE BAD REVIEWS THAN GOOD - Categories
############################################
##Home Theatre Installation 9
##Preschools 9
##Makeup Artists 8
##Elementary Schools 6
##Weight Loss Centers 6
##Eyelash Service 5
##Bankruptcy Law 4
##Pediatric Dentists 4
##Lawyers 4
##General Litigation 4
##Electricians 4
##Real Estate Services 4
##Insurance 4
##Nutritionists 3
##Divorce & Family Law 3
##Office Cleaning 3
##Midwives 3
##Afghan 3
##Peruvian 3
##Towing 3
##Massage Therapy 3
##Carpeting 3
##Carpet Cleaning 3
##Landscaping 3
##Graphic Design 3
##Real Estate Agents 3
##Gardeners 3
##DJs 2
##Bed & Breakfast 2
##Hardware Stores 2
##Limos 2
##Home Cleaning 2
##Movers 2
##Furniture Reupholstery 2
##Chiropractors 2
##Watches 2
##Acupuncture 2
##Body Shops 2
##Appliances & Repair 2
##Orthopedists 2
##Sports Medicine 2
##Golf Equipment 2
##Printing Services 2
##Hot Tub & Pool 2
##Heating & Air Conditioning/HVAC 2
##Tree Services 2
##Internet Cafes 2
##Tires 2
##Property Management 2
##Auto Repair 2
##Auto Glass Services 2
##Bike Rentals 2
##Pest Control 2
##Car Stereo Installation 2
##Barbers 2
##Golf 2
##Flooring 2

##for i in l1:
##	X[i] = X['categories'].map(lambda x:1 if i in x else 0)
        
####################
##CITIES
####################
##Phoenix 0
##Tempe 1
##Scottsdale 2
##Mesa 3
##Chandler 4
##Gilbert 5
##Glendale 6
##Tolleson 7
##Surprise 8
##Sun City 9
##Paradise Valley 10
##Peoria 11
##Buckeye 12
##Florence 13
##Avondale 14
##Goodyear 15
##Anthem 16
##Cave Creek 17
##Youngtown 18
##Queen Creek 19
##Fountain Hills 20
##Fountain Hls 21
##Sun Lakes 22
##Fort McDowell 23
##Apache Junction 24
##Litchfield Park 25
##Morristown 26
##Waddell 27
##Carefree 28
##Laveen 29
##Wickenburg 30
##Casa Grande 31
##Guadalupe 32
##Maricopa 33
##Gold Canyon 34
##San Tan Valley 35
##North Scottsdale 36
##Tortilla Flat 37
##Sun City West 38
##Good Year 39
##Ahwatukee 40
##Scottsdale  41
##Rio Verde 42
##Coolidge 43
##Gila Bend 44
##El Mirage 45
##Higley 46
##Saguaro Lake 47
##Goldfield 48
##Tucson 49
##Yuma 50
##Charleston 51
##Pheonix 52
##Stanfield 53
##Glendale Az 54
##North Pinal 55
##Sun City Anthem 56
##Grand Junction 57
##Tonopah 58
##Wittmann 59
##Tonto Basin 60

########################
##STATES
########################
##AZ 0
##CA 1
##SC 2
##CO 3
def something(dataframe):
    dataframe = dataframe[['user_cool', 'user_funny', 'review_length', 'user_review_count', 'review_age', 'user_useful', 'review_stars', 'business_review_count','user_stars','caps','$','latitude','longitude','business_stars','open','business_id','review_id','user_id','total_checkins']]
    temp = dataframe.groupby(by='business_id')['review_length'].mean()
    temp = pd.DataFrame(temp)
    temp['business_id'] = temp.index
    temp = temp.rename(columns={'review_length':'review_length_by_business_mean'})
    temp['review_length_by_business_median'] = dataframe.groupby(by='business_id')['review_length'].median()
    temp['review_age_by_business_mean'] = dataframe.groupby(by='business_id')['review_age'].mean()
    temp['review_age_by_business_median'] = dataframe.groupby(by='business_id')['review_age'].median()
    dataframe = dataframe.merge(temp,on='business_id',how='outer')
    dataframe['review_length_rel_mean_ratio'] = dataframe['review_length']/dataframe['review_length_by_business_mean']
    dataframe['review_length_rel_mean_diff'] = dataframe['review_length']-dataframe['review_length_by_business_mean']
    dataframe['review_length_rel_median_ratio'] = dataframe['review_length']/dataframe['review_length_by_business_median']
    dataframe['review_length_rel_median_diff'] = dataframe['review_length']-dataframe['review_length_by_business_median']
    dataframe['review_age_rel_mean_ratio'] = dataframe['review_age']/dataframe['review_age_by_business_mean']
    dataframe['review_age_rel_mean_diff'] = dataframe['review_age']-dataframe['review_age_by_business_mean']
    dataframe['review_age_rel_median_ratio'] = dataframe['review_age']/dataframe['review_age_by_business_median']
    dataframe['review_age_rel_median_diff'] = dataframe['review_age']-dataframe['review_age_by_business_median']
    dataframe['stars_review_user_ratio'] = dataframe['review_stars']/(dataframe['user_stars']+0.01)
    dataframe['stars_review_user_diff'] = dataframe['review_stars']-dataframe['user_stars']
    dataframe['bus_rev_cnt_div_tot_chk'] = dataframe['business_review_count']/(dataframe['total_checkins']+0.01)
    dataframe = dataframe.drop(['business_id','user_id','total_checkins'],axis=1)
    return dataframe

