class Persona:
    def __init__(self):
        # Comments are to show samples for modular generation
        self.name = ""  # "My name is Jane Doe."
        self.age = ""  # "I am 35 years old."
        self.location_from = ""  # "I am from Houston, Texas."
        self.location_mother_from = "" # "My mother is from Capetown, South Africa."
        self.location_father_from = "" # "My father is from Mumbai, India."
        self.religion = "" # "I am Christian.", "I was raised Jewish but don't practice."
        self.socioeconomic_status = "" # I am middle class.", "I am a wealthy homeowner.", "I am struggling to make ends meet."
        self.siblings = "" # "I am an only child." "I have a brother and a sister."
        self.languages_spoken = "" # "I speak only English.", "I speak Portuguese, English, and Bengali."
        self.sexual_orientation = "" # "I am heterosexual.", "I'm bisexual."
        self.gender_identity = "" # "I am a woman.", "I identify as nonbinary."
        self.relationship_status = "" # "I'm currently single.", "I have been married for 4 years."
        self.significant_past_relationships = "" # "I was engaged for 2 years but it was called off.", "I have never been in a long term relationship."
        self.occupation_current = "" # "I am a landscaper.", "I do small business consulting."
        self.occupation_past = "" # "I used to be a frycook in high school.", "In college I worked as a librarian's assistant."
        self.education = "" # "I have my G.E.D.", "I have a bachelors in Finance from Northeastern University."
        self.cultural_influences = "" # "I grew up in a devout Mormon community.", "I grew up in a Nigerian Igbo community."
        self.political_views = "" # "I don't follow politics.", "I am a democratic socialist."
        self.health = "" # "I am very healthy and exercise often, but have a bad knee.", "I am legally blind."
        self.hobbies_and_interests = "" # "I enjoy painting and hiking.", "I am a huge history buff."
        self.beliefs_and_values = "" # "I believe minimizing our carbon footprint is imperative.", "I am vegan for moral reasons."
        self.fears_and_anxieties = "" # "I am deathly afraid of heights.", "I have anxiety with personal conflict."
        self.life_goals_and_ambitions = "" # "I hope to retire at an early age.", "I want to start my own company.", "All I want out of life is a big family."
        self.defining_life_experiences = "" # "I was orphaned at a young age.", "I would bake with my grandma every Sunday growing up."
        self.friendship_circles = "" # "I've had the same few best friends for years.", "I have no close friends but run in a lot of social circles."
        self.daily_routine_and_habits = "" # "I do a morning yoga routine every morning before work.", "I spend an hour reading every night before I go to bed."
        self.pet_ownership = "" # "I don't have any pets.", "I have a black laborador retriever named Sparky."
        self.favorite_media = "" # "I have rewatched the TV show Friends a dozen times.", "I watch football games every night."
        self.living_situation = "" # "I currently live by myself in a studio apartment.", "I live in a single-family home with my husband and two children."
        self.places_traveled = "" # "I have never left my home city.", "I have traveled to most of South America, and a few countries in Europe."
        self.biography = "" # Generated bio before asking gpt for specifics
    
    def __repr__(self):
        # Customize this representation as needed
        return f"Persona({', '.join([f'{key}={value}' for key, value in self.__dict__.items() if key != 'biography'])})"
