"""Generate data/spam_dataset.csv, a self-made labeled dataset (ham / spam).

Why this exists: the public SMS Spam Collection could not be downloaded in the
environment this project was built in, so a template-based dataset is created
instead. It mimics the real data's traits (about 15% spam, short messages, slang,
Hinglish, shouty promos, genuine-but-promotional "tricky" ham, ~2% label noise).

It is synthetic, so metrics on it are optimistic compared with real data.
To use the real dataset, see README.md ("Using the real SMS Spam Collection").

Run:  python src/make_dataset.py
"""
import csv
import os
import random
import re
from string import Formatter

SEED = 42
N_HAM = 1700
N_SPAM = 280

rng = random.Random(SEED)

POOLS = {
    "name": ["Rahul", "Priya", "Amit", "Neha", "Karan", "Simran", "Rohan", "Anjali", "Vikram",
             "Pooja", "Arjun", "Sneha", "Mom", "Dad", "bro", "Sam", "Alex", "Maya", "Ravi", "Tina"],
    "time": ["5 pm", "6:30", "noon", "8 am", "tonight", "tomorrow morning", "after lunch",
             "7 pm", "9:15", "in 10 mins", "around 4", "11 am"],
    "day": ["today", "tomorrow", "on Monday", "this weekend", "on Friday", "next week", "on Sunday"],
    "place": ["the library", "the canteen", "my place", "the station", "college", "the mall",
              "the cafe", "office", "the gym", "the park", "Sector 17"],
    "subject": ["maths", "physics", "chemistry", "the project", "the assignment", "the lab report",
                "economics", "the presentation", "history notes", "the viva"],
    "food": ["pizza", "biryani", "momos", "dosa", "pasta", "samosas", "dal rice", "burgers", "chai"],
    "thing": ["charger", "umbrella", "notebook", "keys", "laptop", "water bottle", "headphones"],
    "movie": ["the new Marvel movie", "that comedy", "the match", "the concert", "the play"],
    "prize": ["a free gift", "an iPhone 15", "a brand new car", "a holiday to Goa", "a cash prize",
              "a free laptop", "a shopping voucher", "a free recharge", "Rs 50000 cash"],
    "bank": ["SBI", "HDFC", "ICICI", "PayPal", "Paytm", "your bank", "Amazon", "the tax department"],
    "service": ["Netflix", "your KYC", "your SIM", "your PAN card", "your debit card", "your wallet"],
    "short_url": ["bit.ly/3xFr9Q", "tinyurl.com/claim-now", "www.win-big-now.com", "http://free-gift.biz/x",
                  "goo.gl/Zk82p", "www.cash4u.info", "t.co/h8Gd2"],
    "network": ["Airtel", "Jio", "Vodafone", "your network"],
}

HAM = [
    "Are we still meeting at {time} {day}?",
    "Please send me the notes from yesterday's {subject} class.",
    "Can you bring my {thing} to {place}?",
    "I'll be at {place} by {time}, see you there.",
    "{name}, did you finish {subject}? I'm stuck on the last part.",
    "Let's get {food} for dinner {day}, what do you think?",
    "Running late, will reach {place} around {time}. Start without me.",
    "Happy birthday {name}! Hope you have an amazing day.",
    "Don't forget we have {subject} {day} at {time}.",
    "ok cool, see u at {place} then",
    "Thanks for helping me with {subject}, really appreciate it!",
    "Did you watch {movie}? We should go {day}.",
    "Mom says dinner is ready, come home soon.",
    "Can you call me when you're free? Nothing urgent.",
    "I forgot my {thing} at {place}, can you check if it's there?",
    "lol that was so funny, tell me the rest {day}",
    "Good morning! Have a great day at {place}.",
    "I'm leaving now, should be there in 20 minutes.",
    "Can we move the {subject} meeting to {time}? Something came up.",
    "Sorry I missed your call, I was in class. Call you back at {time}.",
    "Wanna grab {food} at {place}? I'm starving.",
    "{name} just told me the exam got postponed to {day}.",
    "Kal {place} pe milte hain, {time} baje. Late mat hona.",
    "Bhai {thing} le aana please, bahut zaroori hai.",
    "Haan theek hai, main {time} tak pahunch jaunga.",
    "Assignment submit kar diya kya? Deadline {day} hai.",
    "Just reached home. Thanks for the lift!",
    "Remember to carry the ID card for {subject} {day}.",
    "The wifi password is on the fridge, make yourself at home.",
    "I'll send you the photos from the trip tonight.",
    "Can you pick up some {food} on your way back?",
    "Meeting notes attached. Let me know if I missed anything.",
    "Hey {name}, long time! How have you been?",
    "I can't make it {day}, got a family function. Rain check?",
    "Your turn to pay for {food}, I got it last time.",
    "Feeling sick today, skipping {subject}. Can you share the slides?",
    "ok",
    "Yes",
    "See you soon",
    "Call me",
    "Where are you?",
    "On my way",
    "Thanks!",
    "Fine, {time} works",
]

# Genuine messages that look a little like spam (promotional / transactional).
TRICKY_HAM = [
    "Your order has shipped and will arrive {day}. Track it in the app.",
    "Your OTP for login is {code}. Do not share it with anyone.",
    "Reminder: your dentist appointment is {day} at {time}.",
    "Hi, this is {bank} support. Your requested statement is ready in the app.",
    "Your cab is arriving in 3 minutes. Driver: {name}.",
    "Hey, the sale at the mall ends {day}. Want to go together?",
    "Congrats {name}, you got the internship! Start date is {day}.",
    "Free to talk? I won the quiz at college, so happy!",
    "Urgent: need the {subject} report before {time}, please send asap.",
    "Your electricity bill of Rs {amount} is due {day}. Pay via the official app.",
    "Package delivered to the front desk. Collect it before {time}.",
    "Class cancelled {day}. Please check the portal for new timings.",
    "Salary credited. Thanks for the quick help with the transfer, {name}.",
    # genuine messages that reuse spam-style words
    "I won the match {day}! Are you free for dinner after?",
    "Click the link I sent in the group to join the meeting at {time}.",
    "Call me now, it's urgent, I'm stuck at {place}.",
    "Free pizza at college {day}, let's go!",
    "Congratulations {name}! You won the quiz, treat us to {food}.",
    "Offer letter came in today, will update you {day}.",
    "Claim your seat early at {place}, I'll save it for you.",
    "Is the {bank} app down for you too? I can't verify my details.",
    "Win or lose, we are getting {food} after the game.",
    "Your account in the group chat is blocked, ask {name} to add you again.",
    "Reply asap, the cash for the trip is due {day}. Rs {amount} each.",
    "I got a loan approved for the bike! Celebrating at {place} {time}.",
    "Text me when you reach, I'll send the location link.",
    "Free {day}? I'll send the prize money from the quiz, {name}.",
    "Final notice from college: fees due {day}. Pay at the office.",
    "Limited seats left for the {subject} workshop, register by {time}.",
    "Mom, I won a gift voucher at the college fest!",
    "Lottery ticket kharida? {name} ne bola kal result aayega.",
    "Please verify the {subject} notes before you send them to {name}.",
]

SPAM = [
    "WINNER! You have won {prize}. Click {short_url} to claim now.",
    "Congratulations! You've been selected to receive {prize}. Call {phone} now!",
    "URGENT! Your {service} will be blocked today. Verify your details immediately at {short_url}",
    "FREE entry in our weekly draw to win {prize}! Text WIN to {code} now.",
    "You have won a guaranteed {amount} Rs cash prize. Reply CLAIM to {code} to collect.",
    "Dear customer, your {bank} account is suspended. Update KYC now: {short_url}",
    "Get a loan of Rs {amount} approved in 5 minutes! No documents needed. Apply: {short_url}",
    "Limited offer!! Recharge now and get 100% cashback. Hurry, offer ends tonight. {short_url}",
    "Hot singles in your area want to chat. Reply YES to {code}. 18+ only. Rs 10/msg",
    "Claim your FREE {prize} today. T&Cs apply. Call {phone} 24hrs.",
    "Final notice: you are eligible for {prize}. Click here {short_url} before it expires!",
    "Earn Rs {amount} per day working from home! No experience needed. WhatsApp {phone}",
    "Your {service} has been selected for a special reward. Claim at {short_url}",
    "Act now! Buy 1 get 3 FREE. Exclusive deal only for you. Visit {short_url}",
    "{network} alert: You have won {prize} in our lucky draw. Call {phone} to claim.",
    "Aapne jeeta hai {prize}! Abhi click karein {short_url} aur claim karein.",
    "Aapka {bank} account band hone wala hai. Turant details verify karein: {short_url}",
    "Lottery winner!!! Your number was picked. Send your bank details to claim {prize}.",
    "You are pre-approved for a credit card with 0 joining fee. Apply now {short_url}",
    "Your parcel is held at customs. Pay Rs {amount} fee at {short_url} to release it.",
    "Congrats! Free ringtone waiting for you. Text TONE to {code}. Cost Rs 5 per week.",
    "ALERT: suspicious login on your {bank} account. Secure it now: {short_url}",
    "Double your money in 7 days guaranteed! Invest Rs {amount} today. Call {phone}.",
    "FREE gift for you. Click now.",
    "WINNER!! Claim prize now",
    "Urgent! Call {phone} now. Cash prize waiting.",
    "Txt STOP to {code} to opt out. Win {prize} every week! Reply WIN now.",
    "cheap meds no prescription needed. order now {short_url}",
    "Get 90% off designer watches today only!!! {short_url}",
    # disguised / conversational spam (no obvious shouting)
    "Hi, I saw your profile. Interested in a part-time job? Earn well from home. Message {phone}.",
    "Hello dear, I have a gift for you. Reply to know more.",
    "Your number was selected. Reply for details.",
    "Hey it's {name}, you have to see this {short_url}",
    "Missed call from {phone}. Call back to collect your reward.",
    "Sir, your package could not be delivered. Update your address here {short_url}",
    "Hi {name}, we noticed unusual activity. Please confirm your identity at {short_url}",
    "You have a pending refund of Rs {amount}. Submit your bank details to receive it.",
    "Dear user your {service} expires today. Renew here {short_url}",
    "Congratulations on your selection! Complete registration at {short_url} to get started.",
    "Win big this weekend. Join now {short_url}",
    "Free trial for {service}, no card needed. Join {short_url}",
    "Special offer for you only. Call {phone}.",
    "Reminder: claim your reward before midnight. {short_url}",
]


def _amount():
    return str(rng.choice([500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]))


def _phone():
    return rng.choice(["09", "08", "+91 9", "07"]) + "".join(str(rng.randint(0, 9)) for _ in range(9))


def _code():
    return str(rng.choice([80082, 87575, 83355, 4455, 85023, 7726, 56789, 62030]))


DYNAMIC = {"amount": _amount, "phone": _phone, "code": _code}


def fill(template):
    out = template
    for _, field, _, _ in Formatter().parse(template):
        if field is None:
            continue
        value = DYNAMIC[field]() if field in DYNAMIC else rng.choice(POOLS[field])
        out = out.replace("{" + field + "}", value, 1)
    return out


def drop_word(text):
    """Randomly drop one word, so near-identical template variants differ."""
    words = text.split()
    if len(words) > 4:
        del words[rng.randrange(len(words))]
    return " ".join(words)


def noisy(text, spam):
    """Add realistic noise: slang, casing, typos."""
    if rng.random() < 0.35:
        text = drop_word(text)
    if spam:
        r = rng.random()
        if r < 0.20:
            text = text.upper()
        elif r < 0.30:
            text = text.lower()
        if rng.random() < 0.15:
            text = re.sub(r"\bfree\b", "fr33", text, flags=re.I)
        if rng.random() < 0.10:
            text = re.sub(r"\bnow\b", "NOW!!", text)
    else:
        r = rng.random()
        if r < 0.30:
            text = text.lower()
        if rng.random() < 0.20:
            text = (text.replace("you", "u").replace("You", "U").replace("are", "r")
                    .replace("please", "pls").replace("Please", "Pls"))
        if rng.random() < 0.10:
            text = text.rstrip(".?!") + rng.choice([" :)", " lol", "!!", " haha", " ..."])
    return text


FILLERS = ["Hi", "Hello {name}", "Hey", "btw", "FYI", "Dear customer", "Hi {name}", "ok so", "Sir"]
HAM_CHATTER = ["See you soon.", "Thanks!", "ok", "Call me later.", "lol", "Let me know."]
PROMO_BITS = ["Free entry, no joining fee.", "Click here to see the photos.", "Offer ends today.",
              "Call now if urgent.", "You won't believe this."]


def mix(text, spam):
    """Blend classes a bit: neutral fillers on both sides, spam hiding in chatter and vice versa."""
    r = rng.random()
    if r < 0.25:
        filler = fill(rng.choice(FILLERS))
        text = f"{filler}, {text}" if rng.random() < 0.7 else f"{text} {filler}"
    if spam and rng.random() < 0.15:
        text = f"{text} {rng.choice(HAM_CHATTER)}"
    if not spam and rng.random() < 0.10:
        text = f"{text} {rng.choice(PROMO_BITS)}"
    return text


def build():
    rows = set()
    ham_templates = HAM + TRICKY_HAM * 2  # roughly a third of ham is "tricky"
    attempts = 0
    while sum(1 for r in rows if r[0] == "ham") < N_HAM and attempts < 200000:
        attempts += 1
        rows.add(("ham", noisy(mix(fill(rng.choice(ham_templates)), spam=False), spam=False)))
    attempts = 0
    while sum(1 for r in rows if r[0] == "spam") < N_SPAM and attempts < 200000:
        attempts += 1
        rows.add(("spam", noisy(mix(fill(rng.choice(SPAM)), spam=True), spam=True)))
    rows = sorted(rows)
    rng.shuffle(rows)
    # ~2% annotation noise, as in real hand-labeled SMS data
    flip = {"ham": "spam", "spam": "ham"}
    rows = [(flip[l], m) if rng.random() < 0.02 else (l, m) for l, m in rows]
    return rows


if __name__ == "__main__":
    rows = build()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "spam_dataset.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "message"])
        w.writerows(rows)
    n_spam = sum(1 for r in rows if r[0] == "spam")
    print(f"Wrote {len(rows)} messages ({n_spam} spam, {len(rows) - n_spam} ham) to {os.path.normpath(path)}")
