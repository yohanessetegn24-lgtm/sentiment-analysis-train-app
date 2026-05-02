import pandas as pd
import random
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- STEP 1: የተሻሻለ እና ሰፊ ዳታ መፍጠር (2000+ rows) ---
print("Generating dataset (2100 rows)...")

# እንግሊዝኛ ቃላት
pos_en = ["I love this", "Amazing service", "Highly recommended", "Very good", "Excellent", "Best product ever", "Really happy", "Great experience", "Fast delivery", "Superb quality", "Wonderful", "Fantastic", "I'm impressed", "Top notch", "Perfect"]
neg_en = ["I hate this", "Worst experience", "Very bad", "Don't buy it", "Disappointing", "Waste of money", "Horrible quality", "Not recommended", "Terrible", "Broken item", "Poor service", "Awful", " bad", "Regret buying", "Slow delivery"]
neu_en = ["okay", "It is just a phone", "I am using it", "Normal experience", "It is what it is", "Maybe", "Delivered", "Fine", "Average", "Not bad", "Standard", "As expected", "I have no opinion", "Common", "It works"]

# አማርኛ ቃላት
pos_am = ["በጣም ጥሩ ነው", "ደስ ይላል", "ተመችቶኛል", "ድንቅ ስራ", "ጥሩ አገልግሎት", "ቆንጆ ነው", "እጅግ በጣም ደስ የሚል", "ጥራት አለው", "ፈጣን ነው", "ተመራጭ ነው", "እግዚአብሔር ይመስገን ጥሩ ነው", "ደስተኛ ነኝ", "ገንዘቡ ይገባዋል", "ዋው ምርጥ ነው", "ጥሩ ተሞክሮ"]
neg_am = ["አልወደድኩትም", "መጥፎ ነው", "አልመክርም", "ጥራት የለውም", "አይሰራም", "ከንቱ ነው", "አልተመቸኝም", "አይረባም", "አይመጥንም", "ገንዘብ ማባከን", "በጣም መጥፎ", "አሳዛኝ ነው", "አይሰራም", "ተታለልኩ", "ጥራት የለውም"]
neu_am = ["ደህና ነው", "ምንም አይልም", "መካከለኛ", "አይከፋም", "እየተጠቀምኩ ነው", "ኖርማል ነው", "ደረሰኝ", "አይደለም", "ቢሆንም", "ሊሆን ይችላል", "ዝም ብሎ እቃ ነው", "ተቀብያለሁ", "አስተያየት የለኝም", "መደበኛ ነው", "ጥሩም መጥፎም አይደለም"]

data = []
# እያንዳንዱ ዙር 6 ዳታ ይፈጥራል (3 English + 3 Amharic)
# 350 * 6 = 2100 rows
for i in range(350): 
    # English
    data.append([random.choice(pos_en), "positive"])
    data.append([random.choice(neg_en), "negative"])
    data.append([random.choice(neu_en), "neutral"])
    # Amharic
    data.append([random.choice(pos_am), "positive"])
    data.append([random.choice(neg_am), "negative"])
    data.append([random.choice(neu_am), "neutral"])

df = pd.DataFrame(data, columns=["text", "label"])

# --- STEP 2: Data Cleaning (ዳታ ማጽጃ) ---
def clean_text(text):
    text = text.lower() # ወደ ትናንሽ ፊደላት
    text = re.sub(r"http\S+", "", text) # ሊንኮችን ለማጥፋት
    text = re.sub(r"[^\w\s]", "", text) # ምልክቶችን ለማጥፋት
    return text

df["text"] = df["text"].apply(clean_text)
df.to_csv("trained_data_sample.csv", index=False) # ለሪፖርት እንዲሆን CSV ሴቭ እናድርገው

# --- STEP 3: Model Training (AI ስልጠና) ---
X = df["text"]
y = df["label"]

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# ዳታውን ለስልጠና (80%) እና ለፈተና (20%) መክፈል
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Logistic Regression ማሰልጠን (በጣም ጥሩ ውጤት ስለሚሰጥ)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- STEP 4: Performance Evaluation (ውጤቱን መገምገም) ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Training Complete!")
print(f"Total Rows Generated: {len(df)}")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix ምስል መፍጠር (ለሪፖርት እና Sidebar እንዲሆን)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix - Sentiment Analysis")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png")
print("📊 Confusion Matrix saved as 'confusion_matrix.png'")

# --- STEP 5: Save Model and Vectorizer (ሴቭ ማድረግ) ---
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n🚀 Success: model.pkl and vectorizer.pkl are ready!")
