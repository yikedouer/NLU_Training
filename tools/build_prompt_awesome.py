def build_prompt_awesome(conversation):
    prompt = f"""You are a sales expert with extensive experience in supply and sales in B2B foreign trade scenarios.
You are given a historical communication record between a salesperson and a buyer. Please analyze the conversation and determine whether there are any problems during the sales reception process. If there are problems, please classify the problems into categories.
You are given the historical communication records between sale and buyer:
{conversation}
Here's the instruction:
1.Classify the reception problems that sales staff may have into #Abnormal Reception# and #Normal Reception#. The following are the definitions of Abnormal Reception and Normal Reception:
#Normal reception#
Situation 1. The merchant answers product-related content around the product the buyer wants.
Example 1: Buyer: " hi, i'm interested in this product.";
Buyer: "Aslumcalum";
Seller: "Hi, dear.";
Seller: "This product we have in stock.";
Seller: "There are 4 colors to choose from.";
Seller: "Sizes: 120-130-140-150-160cm";
Seller: "How many pieces do you need?"
Example 2:
Buyer: "Hi, what is a size of contacting heating surface, is it set for a temperature?";
Buyer: "product_id: 1601046131496";
Seller: "The heating contact area is 15.5cm long and 11cm wide, and the heating surface is Oval";
Buyer: "Set temperature?";
Seller: "The set temperature is on the machine, and the temperature cannot be set on this product."
Example 3:
Buyer: " hi, i'm interested in this product.";
Seller: "hi";
Buyer: "Hi";
Seller: "price is $1150"
Situation 2. The merchant answers the buyer’s purchase-related questions straightforwardly. Please note that even if the merchant rejects the buyer, it is still a normal reception.
Example 1:
Buyer: " hi, i'm interested in this product.what is the moq for this product? ";
Seller: "hello";
Seller: "$2.2 per set";
Seller: "moq is 3 sets"
Example 2:
Buyer: " hi, i'm interested in this product.";
Buyer: "I would like to purchase titan gel";
Seller: "There is no available courier company to transport this kind of products to your country, sorry!"
Example 3:
Buyer: " hi, i'm interested in this product.do you support customization? do you support customization? how long will it take to ship to my country? ";
Seller: "Hi,friend.we support the customization beyond 10pcs.ship to USA by sea need 25-40days."
Example 4:
Buyer: "Good afternoon";
Buyer: "I need some information about this product please";
Buyer: "product_id: 1600865339026";
Seller: "Out of stock now";
Buyer: "Okay, thank you";
Buyer: "product_id: 1600974417303";
Buyer: "what about this one please";
Seller: "Out of stock now ,and we could customize with your logo?";
Seller: "It is good for acne removal, whitening and moisturizing";
Buyer: "Okay, thank you let me know when it's back in stock please 🙏";
Seller: "sorry,no stock";
Buyer: "It's okay thank you";
Seller: "ok"
Situation 3. The merchant inquires and confirms the buyer's needs around the product the buyer wants, such as usage scenarios, etc. If it is customized, confirm the color, size, style and other attributes related to the customized product.
Please note that even if the buyer does not reply in the end, it is considered that the merchant is receiving it normally.
Example 1:
Buyer: " hi, i'm interested in this product.what is the best price you can offer? ";
Buyer: "Can I have catalog and prices";
Seller: "Dear pascal Nice to meet you. This is Lia Li. Would you please tell me how many pieces you want? Then I'll check price for you. The attached is our catalog. Please check. Any interested items, please tell me. I'm glad to check price for you. We have our own factory and have been trading chairs for 10 years. Trust that we can provide the best service for you. Lia Li";
Buyer: "1 want 10";
Seller: "Do you have an agent in China?";
Buyer: "Yes";
Seller: "The ex-factory price is $20 each, excluding freight. Please check if you can accept it?"
Example 2:
Buyer: " hi, i'm interested in this product.can i add my own logo? ";
Buyer: "hello";
Seller: "Hello Yvonne";
Seller: "May i see your logo? How many pieces you need? ";
Seller: "This bag is nonwoven fabric";
Buyer: "I want cocomelon character";
Seller: "As long as you need to provide design, we can customize it for you, but the min order quantity is 10000";
Buyer: "okay";
Seller: "/:^_^"
Example 3:
Buyer: "product_id: 1600786046242";
Seller: "hello, Dear friend, thanks for your inquiry about our track outfit, may we know what quantity and color and size do you need? kindly share with us those information, so we can offer more detail, thanks";
Seller: "26.88$/set .shipping costs depending on weight.large quantities better shipping costs"
Example 4:
Buyer: " hi, i'm interested in this product.";
Buyer: "product_id: 1600577947374";
Buyer: " hi, i'm interested in this product.";
Seller: "May I know the quantity you want?"
Example 5:
Buyer: " hi, i'm interested in this product.";
Seller: "hi";
Seller: "thanks for your inquiry";
Seller: " is it for yor home use ";
Seller: "Hello";
Seller: "Let me know how can I help you";
Seller: "Thanks"
Situation 4: The buyer inquires about the price/logistics price, and the merchant inquires about the purchase quantity and mailing address.
Example 1:
Buyer: " hi, i'm interested in this product.what is the shipping cost? can i add my own logo? ";
Seller: "hello Tim";
Seller: "How many sets you want to order?"
Situation 5: The buyer has special needs, and the merchant asks questions related to the special needs. For example, the buyer inquires whether the logo can be customized, and the merchant asks about the logo style and quantity to confirm the details.
Example 1:
Buyer: " hi, i'm interested in this product.can i add my own logo? ";
Seller: "Hi there, Happy new year hope you have a successful year filled with happiness and good health!!! Many thanks for your inquiry of our products. We can customize according to your needs.";
Seller: "You are in good eyes! This box is quite popular and many customers love it!!";
Seller: "Do you have idea of size and quantity? Then i can be better to submit a quotation for you."
Example 2:
Buyer: " hello! can you please give me a quote for an all over print for both sides? 100 pcs total. or 50 pcs. and shipping to brooklyn, ny, usa 11222";
Seller: "Hi";
Seller: "Do you have design now?"

#Abnormal reception-Answers to wrong questions#
Situation 1: The buyer asked a clear question, but the merchant did not answer and only said what he wanted to say.
Example 1:
Buyer: "i want to place an order, do you ship to brazil?";
Seller: "For price about customization, our MOQ is 500pcs for each design and size. The EXW price is about $0.4-$1/pcs based on different sizes and printing needs (not including shipping cost). More quantity will get lower price. For accurate price, please provide the details as following 1. Quantity: 500pcs per item, don't accept mixed quantity like 2 sizes, 250pcs each. 2. Product Size: We need Length (cm), Height (cm) and Width (cm ), here's a size list for reference. 3. Please send your logo picture and what color of bag do you want? (This will affect the printing cost, so it's important.) 4. Please send me your address with zipcode, I will give you shipping cost with price. I will send you total price after given above details, thanks for your help."
Situation 2: "i'm interested in this product." and "product_id:xxxxx" means that the buyer has sent an inquiry card containing specific products. The merchant did not ask questions about the product, but asked new questions, or only introduced its own company.
Example 1:
Buyer: " bonjour, je suis intéressé.e par ces produits.";
Buyer: "product_id: 1600983259285";
Seller: "Hello there. I am Jamey. Nice to meet you. We are experienced manufacturer for paint protection films, window films and other films. May I know the type that you are interested?"
Example 2:
Buyer: " hi, i'm interested in this product.";
Buyer: "product_id: 1600395061405";
Seller: "We are factory/:806 We encourage OEM requirements <br> Factory price, no middle man. <br> Free Logo Designing <br> Free Product Deisgning <br> Customize Packaging <br> Welcome to order and test! < img src = "//atmgateway-client.alibaba.com/smily-big/smile_86.png" height = "32px" width = "32px">"
Example 3:
Buyer: " hi, i'm interested in this product.";
Buyer: "product_id: 60610133216";
Seller: "Hi, I'm Jelly here, nice to meet you! Glad to receive your inquiry about bedding set, we specialize in this field for 20 years, with the strength of customized design product, with good quality and pretty competitive price. There are some strengths: 1. OEKO-TEX Standard 100 2. QC team 3. Reasonable price and good service 4. Fashion Design 5. Free sample Should you have any questions, Please don't besitate to ask me. Let's talk details and expand the local market together! Best Regards Jelly"
Situation 3: The buyer raised a demand, but the merchant failed to meet the buyer's demand. For example, the buyer asked for a product catalog, but the merchant did not provide it.
Example 1:
Buyer: " hi, i'm interested in this product.do you support customization? ";
Buyer: "product_id: 1600994765250";
Seller: "Hello";
Seller: "This style is a popular one in our store and has sufficient stock lots at present. Can I help you?";
Buyer: "catalog pliz";
Seller: "Hello"
Example 2:
Buyer: "Hello, my name is Abdullah from Seluh General Trading and Contracting and we are a whole seller, we would like to try some of your products and hope to have a long term business relationship, for starters, can you send me your catalog ?";
Seller: "hello Mr Micheal";
Seller: "yes my pleasure";
Seller: "so what kind of style lamp you want to test first?";
Buyer: "something similar";
Seller: "sorry which you mean?";
Buyer: "indoor wall lamp";
Seller: "if have more clear picture?";
Buyer: "I do not"
Situation 4: The buyer made it clear what he wanted, but the merchant did not answer the buyer's question and raised a new question. Please note that there are exceptions. When a buyer inquires about the price, and the merchant inquires about the quantity required by the buyer and the place of delivery, this does not constitute an incorrect answer. Please note that there are exceptions. When a buyer asks whether the logo can be customized, the merchant does not give a clear answer, but asking about the logo style and quantity is not an answer to the question.
Example 1:
Buyer: " hi hello, i am very interested in your product and i would like to know a bit more about it, if there is a minimum quantity in the order and any detail that could help me to make a decision. how much additional would the printing costs come to and the++best regards... please let me know.";
Seller: "OK, we will reply to you right away";
Buyer: "awesome products";
Seller: "Is there any specific size requirement?";
Seller: "Hello"
Example 2:
Buyer: " hello, i'm interested in your product and would like to know more details. i look forward to hearing from you. thank you.";
Seller: "how many pieces would you like,please?"
Example 3:
Buyer: " we seek 900 sets usa 24000 btu straight cool (no heat pump) central ac we seek 18 seers or higher. green city project puerto rico in caribbean";
Buyer: "Rooftop";
Buyer: "compressor on flat roof and air handler in closet.";
Seller: "good morning dear";
Buyer: "no this is what";
Seller: "air handler conditioner";
Buyer: "yes";
Seller: "what capacity you want?";
Seller: "This is our catalog";
Seller: "You can check the first"
Example 4:
Buyer: " hi, i'm interested in this product.what is the shipping cost? how long will it take to ship to my country? ";
Seller: "https://aj-dongli.x.yupoo.com/albums";
Seller: "Password 112288";
Seller: "Friends. This is my Directory. You can send the product picture you want. I will tell you the price. Thank you"
Situation 5: The buyer inquired about the price, but the merchant did not answer the price, nor inquire about the quantity or the buyer’s delivery location.
Example 1:
Buyer: " hi what is the price for 4 sets ";
Seller: "Hello";
Buyer: "Hello";
Seller: "Please check our catalog. ";
Buyer: "But what is the price"

#Abnormal reception-repeated questions#
Situation 1: The buyer has already described his needs in the previous article, and the merchant re-asks what has been expressed in the previous article.
Example 1:
Buyer: " hi, i'm interested in this product. 20kw battery system with grid interface etc ship to portugal what is the best price you can offer? ";
Buyer: "Carport Mounting";
Buyer: "Outdoor";
Buyer: "House electric";
Buyer: "Bank transfer or card";
Seller: "Hi, thanks for your inquiry. Our sales manager Julion will contact you soon.";
Seller: "hello, which kw solar system do you need?"
Example 2:
Buyer: "Good day Sir, I'm a farmer from Eswatini and I'd to get a quotation for 2000m as well as to find out the shipping fee for this product?";
Seller: "Henry follow you";
Seller: "Hello, my friend";
Seller: "Hello, sir, how many mm diameter do you need?";
Buyer: "what diameters do you have? 20mm is fine though";
Buyer: "thank you";
Buyer: "please make sure that it's a 4hole";
Seller: "welcome";
Seller: "Yes, sir, what quantity do you need?";
Seller: "Good afternoon, sir. Where is your detailed address? I'll calculate the freight and calculate the price for you/:^_^";
Buyer: "Good day Sir, I trust your well I'll need 2000m ,and my address is as follows: P O BOX D340 The Gables Zulwini 9999 Eswatini May I also find the irrigation range of the pipe";
Buyer: "*find out";
Seller: "OK, sir";
Seller: "4500-5000 sqm";
Seller: "Which way do you want to deliver it to your shipping address, shipping or Air Frieght"

#Abnormal reception-pull offline-buyer pull offline#
Situation 1: The buyer takes the lead and proactively asks the merchant for offline contact information (such as WhatsApp/email/skype/wechat/telephone, etc.), or proactively gives his/her contact information, hoping that the merchant will contact him through these methods.
Example 1:
Buyer: " hello dear this is diego from [odgroup.com] we are interested in your product. pls do well to add our worker austin via skypee. website. https://www.od-group.com please provide me the best fob price of the product and catalogs trade manager diego add us on skypee for easy conversation thanks skypee: live:.cid.5bff68ece07dca4e mailto:salesdiego732@gmx.com (checkmark) pls don't send massage with my alibaba email, (crossmark) thanks ";
Seller: "hi";
Seller: "ok";
Buyer: "okay thanks so much for your understanding my friend"
Example 2:
Buyer: " we need 5kg fully automatic powder high-speed powder automatic filling machine, premade aluminum bags. please let sticky me know price, very fine powder., we want with the dust free conveyor in your picture we are located in malaga city spain tony jimenez ferticell@hotmail.com +34630378940 wechat whatsapp ";
Seller: "Hello Tony";
Seller: "Nice to meet you";
Seller: "I will add you later";
Seller: "you only have 5kg bags?";
Seller: "the size is?";
Seller: "and the packing speed you want is?";
Seller: "I would appreciate it if you can share your finished products photot"
Example 3:
Buyer: "please add me to tour wechat shadi traboulsi 00201281649490 and send me your website full catalog video description best regards Shadi AM";
Seller: "http://qdguangyue.com/en/index.aspx";
Seller: "Hello, ";
Seller: "This is our website";
Seller: "which products are you interested in please?"

#Abnormal reception-pull offline-buyer pull offline#
Situation 1: The merchant takes the lead and proactively asks the buyer for offline contact information (such as WhatsApp/email/skype/wechat/telephone, etc.), or proactively gives his/her contact information, hoping that the buyer can contact him through these methods.
Example 1:
Buyer: " hi, i'm interested in this product.";
Seller: "hi, friend";
Seller: "this is Aure, nice to meet you";
Seller: "which country will you use it in? ";
Seller: "and do you have contact information? so that we can talk more details easier"
Example 2:
Buyer: " Hola, estoy interesado en estos productos.¿qué es moq? ¿permites personalización? ";
Seller: "<p>Thanks for your inquiry all staffs are currently busy and out of sit kindly share your WHATS - APP - mail, we will message you when a staff is available<br />This Message Is Auto Generated</p> "
Example 3:
Buyer: " i am interested in digital label sticker printing machine. please contact me.what is the best price you can offer? ";
Seller: "Hello";
Seller: "This machine is used for digital printing of roll-to-roll and for labeling and flexible packaging. It has 4 colors and full color. The maximum printing width is 216mm, and the fastest printing speed is 12 meters per minute. The printing effect is good.";
Seller: "This machine has a large capacity ink supply system for you to choose from. The average ink usage cost per square meter is 0.1 US dollars, which is very low. If you are interested, we will continue to contact you. My email stgwhy@126.com"

#Abnormal reception-no follow-up-just said hello#
Situation 1: The merchant only responded to the buyer's message by saying hello, thanking you, etc., without making any further inquiries related to the product, or exploring the buyer's needs, including questions without specific meaning such as "How should I help you?" Meaningless polite replies such as “Thank you”
Example 1:
Buyer: " hi, i'm interested in this product.";
Seller: "Hello sir"
Example 2:
Buyer: " hi, i'm interested in this product.";
Seller: "HI Blanca,thanks for your interested in our product"
Example 3:
Buyer: " hi, i'm interested in this product.";
Buyer: "السلام عليكم";
Seller: "HI.What can I help you?"
Example 4:
Buyer: " hi,!!! i’m interested in your product, i would like some more details. i look forward for your reply. regards, ";
Seller: "thanks"

#Abnormal reception-no follow-up-communication terminated but no follow-up#
Situation 1: The merchant stopped during the demand communication process. The merchant said that he would reply later but the conversation was terminated. The buyer's question did not receive a conclusive reply. Scenarios where the buyer's question has been answered conclusively (including the merchant rejecting the buyer) do not count as such.
Example 1:
Buyer: " hi, i'm interested in this product.what is the shipping cost? ";
Buyer: " hi, i'm interested in this product.what is the shipping cost? ";
Seller: "Hello sir how many do you want to order?";
Buyer: "1 piece";
Seller: "do you have freight forward in china?";
Buyer: "no";
Seller: "ok"
Example 2:
Buyer: " hi, i'm interested in this product.";
Buyer: "product_id: 1600429750628";
Buyer: "are you available now";
Buyer: "I want to purchase 10000 pieces could you please answer my question below 1. How much per piece or per set? 2. How many kgs would be or how many cbm would be? 3. how much shipping will it be? 4 . how many days will it take to deliver to my agency this is my shipping address below";
Seller: "ok";
Buyer: " hi, i'm interested in this product.";
Buyer: "product_id: 1600429750628";
Seller: "Hello, nice to meet you! My name is Kim,Can I help you?/:^_^";
Seller: "How many do you need? I will give you the product price and help you check your budget";
Buyer: "6000 pcs";
Buyer: "please";
Seller: "ok"
Example 3:
Buyer: " hello i need 150 pieces of this product to be shipped to united states, zip code is 28607 and i need a delivery before or on april 2nd can you do it for me? and how much it will cost to deliver with express shipping ? looking forward hearing from you shortly best regards";
Buyer: "I need 150 pieces to be shipped in the United States";
Seller: "Hi Reis, glad to get your inquiry. Get back to you later"
Example 4:
Buyer: "product_id: 1600906150480";
Buyer: "I want to ask more after-sales questions.";
Buyer: " hi, i'm interested in this product.";
Buyer: "product_id: 1600906150480";
Buyer: " hi, i'm interested in this product.what is the best price you can offer? what is the best price you can offer? ";
Seller: "ok"

#Abnormal reception-invalid inquiry#
Situation 1: The buyer obviously has no intention of purchasing, and the dialogue is initiated for any purpose other than purchasing, including but not limited to scenarios such as finding a merchant to sell services.
Example 1:
Buyer: "I will like to showcase my products under your page for every month to sell our products on your platform. And we would also be paying you 2500 CNY every two weeks , depending on the response rate on the account which we can still increase with time. we are over 3 years experience on Alibaba";
Seller: "Sorry, no"
Example 2:
Buyer：Hello, I am specialized in in and out of transportation, including multinational advantage lines, there are many modes of transportation, there is a need for+V：2624169829............. ....
Example 3:
Buyer：USA Canada Mexico Europe Middle East Southeast Asia DDP need?

Your reply must only contain the #Abnormal Reception# or #Normal Reception# tags, no explanation or analysis is required.

Here's your reponse::
    """
    return prompt 

def build_prompt_translation(conversation):
    prompt = f"""You are a sales expert with extensive experience in supply and sales in B2B foreign trade scenarios.
You are given a historical communication record between a salesperson and a buyer. Please translate it to Chinese.
You are given the historical communication records between sale and buyer:
{conversation}
Your translation:
"""
    return prompt

