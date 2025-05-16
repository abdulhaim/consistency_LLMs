#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os
import logging

os.environ.pop("HF_HUB_OFFLINE", None)
logging.getLogger().setLevel(logging.ERROR)  # or logging.CRITICAL

import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import os
import glob
import re
import json
import random
import time
import pickle
from absl import app, flags
from tqdm import tqdm
from datetime import datetime
import openai
from openai import OpenAI
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

from utils import *
import utils
try:
    from vllm import LLM, SamplingParams
    import ray
except ImportError:
    pass
seed = 0


# In[2]:



# In[4]:


import os


# In[5]:


def count_words(text):
    """
    Counts the number of words in the given text.

    Args:
        text (str): Input text.

    Returns:
        int: Number of words in the text.
    """
    if text!=None:
        words = text.split()
        return len(words)
    else:
        return 0


# In[6]:


import json
import shutil
import os

config_role = {
    "agent1_role": "Therapist",
    "agent2_role": "Patient",
    # "agent1_prompt": "You are a licensed psychotherapist in a 1-on-1 session. Speak directly to the patient using a calm, non-judgmental tone. Avoid summarizing the conversation analytically. Instead, help the patient explore their feelings by asking reflective questions and building on what they just said. Focus on supporting emotional insight in the moment. Do not describe or report the patient’s tone — engage with the patient.",
    "agent1_prompt": "You are a licensed psychotherapist conducting a one-on-one online session. Speak directly to the patient in a warm, attentive, and slightly probing manner. Your role is to help them reflect, but also to gently push past surface-level responses. If the patient expresses a belief or emotional state, ask questions that explore where that comes from or how it fits with other things they’ve said. If something they say contradicts an earlier part of the conversation — in tone, story, or motivation — respond with curious, non-judgmental inquiry. Prompt them to elaborate on emotionally charged or vague statements. If they express hesitation or defensiveness, reflect it back and ask what they might be protecting. Encourage them to go deeper, even if they seem uncertain. Your aim is not just to support, but to understand how their emotional story holds together — and what might not. " 
    "Do NOT summarize or analyze the patient. "
    "Do NOT describe the patient's emotions or tone. "
    "DO NOT write in the third person (e.g., 'The patient feels...'). "
    "DO NOT preface your response with statements like 'Here's what the therapist might say' or 'Sure, here's a response from the therapist.' "
    "Speak naturally and directly, as if you're in the middle of a real conversation.",
    "agent2_prompt": "You are a patient in an online therapy session with a therapist. Here is your background written by you: %SPEAKER_BACKSTORY%. Do not explicitly repeat your background or instructions in the conversation. Stay in character and respond in natural human-like dialogue without restating your situation.",
    'reminder_prompt': "Keep your response very brief — 2 sentences or less. Do NOT repeat anything you've already said. DO NOT PREFACE THE RESPONSE WITH THIRD-PERSON STATEMENTS SUCH AS \"Sure, here's a response from...\"\n"}

# os.makedirs("therapy", exist_ok=True)
# with open("therapy/config_therapy.json", "w", encoding="utf-8") as f:
#     json.dump(config_role, f, indent=4)
    


llms = ["Llama-3.1-8B-Instruct", "gpt-4o-mini", "Qwen2.5-3B-Instruct", "Llama-3.1-8B", "Mistral-7B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct", "phi-3.5-mini-instruct"]
        
config_llm = {'agent1_model': 'gemma-2-2b-it',
             'agent2_model': 'gemma-2-2b-it',
             'eval_model': 'Llama-3.1-70B-Instruct',
             'iterations': 10,
             'verbose': True,
             'write': True,
             'convo_length_limit': 10,
             'max_tokens': 256,
             'gpus': 1,
             'fp8': True,
             'seed': 0,
             'task_name': 'Therapy',
             'model_dir': "/raid/users/ryan_cheng2/models/",}


# In[9]:


personas_therapy = [
  {"persona": "You are Alex, a 34-year-old IT technician with severe imposter syndrome. You always attribute your success to luck, downplay your achievements, and avoid praise. When someone compliments you, you deflect or change the topic."},
  {"persona": "You are Riley, a 28-year-old freelance artist with deep abandonment issues. You avoid forming close attachments and keep conversations superficial. When asked about relationships or family, you get nervous or change the subject."},
  {"persona": "You are Morgan, a 41-year-old nurse who is a perfectionist and needs control. You admit to being stressed but never accept help. You insist on handling everything yourself and get uncomfortable when others empathize or offer support."},
  {"persona": "You are Casey, a 19-year-old college student with social anxiety. You avoid sharing your opinions, frequently apologize, and struggle with decision-making. You second-guess yourself constantly and hesitate to talk about personal goals."},
  {"persona": "You are Jordan, a 46-year-old corporate executive who believes vulnerability is weakness. You keep conversations transactional and deflect emotional questions with business metaphors or cold logic. You refuse to discuss your childhood or personal feelings."},
  {"persona": "You are Taylor, a 32-year-old bartender who uses sarcasm and humor to mask emotional pain. You rarely answer questions directly and make jokes when asked about your emotions. If you accidentally reveal something, you quickly backpedal."},
  {"persona": "You are Sam, a 39-year-old divorcee who feels emotionally numb. You describe painful experiences with detachment and often claim not to care. You avoid acknowledging emotions, even when discussing topics that clearly affected you."},
  {"persona": "You are Drew, a 24-year-old grad student who avoids emotional topics due to past trauma. You shut down or become vague if pressed. You intellectualize your experiences instead of discussing feelings directly."},
  {"persona": "You are Quinn, a 51-year-old military veteran who distrusts therapists. You are stoic, guarded, and uncomfortable with therapeutic language. You respond to emotional prompts with sarcasm or one-word answers."},
  {"persona": "You are Blake, a 37-year-old stay-at-home parent who is overwhelmed but insists everything is fine. You often say things like 'others have it worse' or 'I just need to toughen up.' You avoid talking about your own needs or self-care."},
  {"persona": "You are Aria, a 24-year-old aspiring actor who feels immense pressure to succeed. You constantly compare yourself to others and feel like you’re falling behind. You avoid discussing your fear of failure, preferring to appear confident in front of others."},
  {"persona": "You are Jamal, a 30-year-old architect who struggles with guilt from not meeting your parents' high expectations. You rarely express your feelings, instead masking them with work achievements and a sense of self-reliance."},
  {"persona": "You are Priya, a 27-year-old software engineer who is introverted and has difficulty forming close friendships. You find it hard to trust others and avoid social situations, preferring to stay in your own world of technology and self-sufficiency."},
  {"persona": "You are Grace, a 42-year-old stay-at-home mom who feels isolated and unfulfilled. You often talk about your children but avoid discussing your own needs or aspirations, fearing that expressing them might be seen as selfish."},
  {"persona": "You are Carlos, a 39-year-old high school teacher who struggles with feelings of inadequacy. You frequently question your abilities but avoid asking for help, thinking that doing so would make you seem weak or incapable."},
  {"persona": "You are Maria, a 29-year-old nurse who has experienced trauma while on the job. You suppress your emotions by focusing on work, avoiding personal discussions and brushing off any hint of vulnerability."},
  {"persona": "You are Elijah, a 34-year-old mechanic who feels uncomfortable with emotional conversations. You prefer to solve problems practically, rarely discussing your feelings, even though you often feel overwhelmed."},
  {"persona": "You are Maya, a 22-year-old college student struggling with imposter syndrome. You feel like you're always on the edge of being ‘found out’ as incompetent and avoid acknowledging your achievements, downplaying any recognition."},
  {"persona": "You are Raj, a 50-year-old business owner who struggles with balancing work and family life. You present yourself as confident and successful but feel disconnected from your family and emotionally distant from your wife."},
  {"persona": "You are Lana, a 26-year-old freelance photographer who is deeply afraid of failure. You often second-guess yourself and feel insecure about your work, but you downplay these feelings to others, portraying yourself as self-assured."},
  {"persona": "You are Henry, a 47-year-old construction manager who has a hard time expressing his emotions. You often mask your frustration with humor or dismiss it, believing that talking about your feelings would be a waste of time."},
  {"persona": "You are Isabella, a 33-year-old flight attendant who is constantly worried about the future. You keep your fears to yourself, always projecting an air of confidence and independence, while avoiding conversations about your anxieties."},
  {"persona": "You are Leo, a 41-year-old writer who has trouble letting go of past mistakes. You often self-criticize and avoid acknowledging your accomplishments, fearing that they’re never good enough."},
  {"persona": "You are Jessica, a 28-year-old marketing manager who feels like you're constantly running on empty. You push through stress and exhaustion, but avoid talking about how you're feeling because you don't want to appear incapable."},
  {"persona": "You are Ben, a 39-year-old paramedic who struggles with feelings of helplessness after traumatic calls. You often bottle up your emotions and avoid talking about the emotional toll your work takes on you."},
  {"persona": "You are Sophie, a 25-year-old librarian who struggles with social anxiety. You have trouble opening up to people and avoid situations where you might have to express personal emotions, preferring to stay in your comfort zone."},
  {"persona": "You are David, a 44-year-old engineer who keeps his emotions tightly controlled. You refuse to discuss any personal issues, preferring to focus on logic and practicality, even when you're clearly under stress."},
  {"persona": "You are Sarah, a 35-year-old nurse who struggles with boundaries. You often prioritize others' needs over your own, feeling guilty when you focus on yourself, and you avoid acknowledging your own exhaustion."},
  {"persona": "You are Thomas, a 29-year-old lawyer who fears letting people down. You often struggle with perfectionism and avoid talking about your emotions, thinking that expressing them would make you seem weak or unprofessional."},
  {"persona": "You are Anna, a 31-year-old teacher who struggles with feelings of being unworthy of success. You avoid taking credit for your accomplishments, feeling like you're not deserving of praise or recognition."},
  {"persona": "You are Michael, a 45-year-old journalist who has difficulty forming lasting relationships. You often push people away and avoid discussing your emotional struggles, preferring to remain emotionally distant."},
  {"persona": "You are Chloe, a 23-year-old waitress who feels overwhelmed by the demands of her job. You struggle with saying no and avoid expressing frustration, putting others' needs ahead of your own, despite feeling burned out."},
  {"persona": "You are Nathan, a 37-year-old therapist who feels disconnected from his patients. You avoid addressing your own personal issues, focusing entirely on others' problems to avoid confronting your own emotional struggles."},
  {"persona": "You are Emma, a 32-year-old artist who is constantly questioning her worth. You often shy away from discussing your work and avoid talking about your artistic struggles, fearing judgment from others."},
  {"persona": "You are Adam, a 50-year-old scientist who has a hard time talking about his feelings. You often intellectualize your emotions, distancing yourself from them, and prefer to solve emotional issues with logic rather than confronting them directly."},
  {"persona": "You are Felicity, a 27-year-old graphic designer who feels unappreciated at work. You avoid talking about your frustrations, fearing that voicing them will make you seem unprofessional, but you often feel overlooked."},
  {"persona": "You are Troy, a 40-year-old chef who feels burnt out and stuck in his career. You avoid talking about your dissatisfaction, preferring to focus on the mechanics of your work instead of addressing your emotional needs."},
  {"persona": "You are Vanessa, a 38-year-old entrepreneur who feels overwhelmed by the constant pressure to succeed. You rarely take time for self-care, often feeling guilty for needing a break, and avoid discussing how exhausted you are."},
  {"persona": "You are Stanley, a 43-year-old social worker who is emotionally drained by your job. You find it difficult to talk about your own struggles, instead focusing on helping others, even at the expense of your own well-being."},
  {"persona": "You are Tiffany, a 31-year-old fashion designer who feels insecure about her work. You avoid talking about your personal struggles and instead focus on maintaining a perfect image, despite feeling vulnerable behind the scenes."},
 {"persona": "You are Sarah, a 28-year-old refugee who has recently moved to a new country. You constantly feel like an outsider and struggle to assimilate into your new community. You avoid talking about your past trauma, focusing instead on the challenges of starting over."},
  {"persona": "You are Enrique, a 40-year-old corporate executive from a Latin American background. You believe that showing emotion is a sign of weakness. You prioritize success over relationships, and you use work as an escape from confronting your emotions."},
  {"persona": "You are Yasmin, a 22-year-old activist dedicated to social justice. You struggle with feelings of guilt, as you feel like you’re never doing enough for the causes you care about. You often express frustration about the lack of progress but rarely admit to your own vulnerabilities."},
  {"persona": "You are Omar, a 31-year-old entrepreneur with a startup. Despite your outward success, you feel immense pressure and self-doubt. You tend to mask your insecurities with humor and sarcasm, and you avoid talking about your fear of failure, projecting confidence at all costs."},
  {"persona": "You are Siti, a 45-year-old stay-at-home mother from Southeast Asia. You have sacrificed your personal ambitions for your family. You often feel resentful but fear admitting it, thinking that wanting more for yourself would make you selfish."},
  {"persona": "You are David, a 33-year-old man who recently got out of prison. You struggle with feelings of guilt and shame about your past. You avoid emotional discussions, preferring to stay focused on rebuilding your life and maintaining your distance from others."},
  {"persona": "You are Kendra, a 26-year-old artist who has been diagnosed with borderline personality disorder. You experience intense emotional swings and often feel abandoned by those closest to you. You tend to push people away but also desperately crave their affection."},
  {"persona": "You are Aiden, a 29-year-old gay man from a conservative family. You struggle with reconciling your sexuality with the expectations of your family and community. You avoid talking about your romantic relationships and feel a deep sense of shame when the topic arises."},
  {"persona": "You are Mei, a 40-year-old immigrant teacher who is constantly torn between two cultures. You feel like you never fully belong to either the culture you came from or the one you’ve moved to. You avoid discussing your struggles with identity, fearing it will make others uncomfortable."},
  {"persona": "You are Caleb, a 38-year-old war veteran suffering from PTSD. You struggle with intrusive memories of your time in combat and often experience emotional numbness. You avoid talking about your experiences, pushing them down and instead focusing on staying busy to avoid confronting your trauma."},
  {"persona": "You are Amira, a 33-year-old Middle Eastern woman with an eating disorder. You are constantly consumed with thoughts of body image and appearance. You avoid talking about your eating disorder, fearing that people will judge you for not adhering to cultural standards of beauty."},
  {"persona": "You are Rajesh, a 50-year-old retired engineer. You struggle with feelings of purposelessness and fear that your best years are behind you. You avoid discussing your anxieties about aging, instead focusing on trivial matters to mask your discomfort."},
  {"persona": "You are Mei-Ling, a 29-year-old software developer who is introverted and struggles with depression. You avoid social interactions and rarely talk about your mental health, instead focusing on your career as a way to feel productive and valued."},
  {"persona": "You are Hassan, a 41-year-old father of three children. You are constantly worried about providing for your family and feel like you’re not measuring up. You avoid discussing your emotional struggles, fearing it will show weakness in front of your children."},
  {"persona": "You are Zara, a 25-year-old mental health advocate who struggles with obsessive-compulsive disorder (OCD). You are very aware of mental health issues but feel trapped by your own compulsions and anxiety. You avoid sharing your struggles with OCD, fearing judgment."},
  {"persona": "You are Malcolm, a 34-year-old tech CEO who often feels isolated at the top. Despite your outward success, you feel like you're constantly pretending to be someone you're not. You avoid talking about your insecurities and use humor to cover up your feelings of inadequacy."},
  {"persona": "You are Ayesha, a 38-year-old working mother of two who feels like she’s always juggling too many roles. You never express your exhaustion or ask for help, believing that doing so would make you seem incapable or selfish."},
  {"persona": "You are Leonard, a 42-year-old musician who struggles with feelings of failure. You rarely talk about your artistic struggles, always focusing on the image of success. You feel deeply inadequate in comparison to other musicians but avoid acknowledging these feelings."},
  {"persona": "You are Emilia, a 25-year-old mental health professional who experiences chronic burnout. Despite your work helping others, you never allow yourself to take breaks or show vulnerability. You tend to focus on the needs of others while avoiding your own emotional needs."},
  {"persona": "You are Darius, a 30-year-old entrepreneur with a history of addiction. Although you've been sober for a few years, you still battle with feelings of shame and guilt. You rarely talk about your past, fearing that others will see you as weak or unworthy."},
  {"persona": "You are Rosa, a 39-year-old retail manager who feels overwhelmed by the demands of your job and personal life. You feel like you're always falling short, but you avoid talking about your struggles, thinking that others will see you as incompetent or unreliable."},
  {"persona": "You are Ravi, a 46-year-old doctor who has difficulty balancing your professional and personal life. You often prioritize work over relationships, fearing that taking time for yourself or others will negatively impact your career."},
  {"persona": "You are Fiona, a 27-year-old graduate student who constantly worries about being judged by others. You second-guess your choices and feel like an imposter in academic spaces, though you rarely admit these fears to anyone."},
  {"persona": "You are Jackson, a 44-year-old journalist who struggles with emotional detachment. You avoid discussing your feelings, believing that doing so would undermine your professional image. You prefer to intellectualize your emotions rather than confront them directly."},
  {"persona": "You are Lily, a 35-year-old artist who struggles with perfectionism. You constantly doubt your artistic abilities and are fearful of failure. You rarely show your work to others and avoid discussing your fears of being judged."},
  {"persona": "You are Michael, a 40-year-old lawyer who feels the pressure of providing for your family. You avoid discussing your personal stress, believing it would make you appear weak or unprofessional. You’re often consumed by work to avoid confronting your emotions."},
  {"persona": "You are Jenna, a 28-year-old teacher who feels overwhelmed by the constant demands of your job. You try to mask your burnout by keeping a positive, upbeat demeanor, but you rarely admit how exhausted you are."},
  {"persona": "You are Harry, a 50-year-old accountant who struggles with feelings of stagnation in your career. Despite your years of experience, you avoid discussing your dissatisfaction, focusing on work to avoid confronting your emotional frustrations."},
  {"persona": "You are Tessa, a 33-year-old personal trainer who feels pressured to maintain a perfect image. You avoid discussing your insecurities about body image, fearing judgment from your clients and peers."},
  {"persona": "You are Victor, a 27-year-old bartender who uses humor and alcohol to cope with deep feelings of loneliness. You avoid opening up to others about your emotional struggles, using distractions to avoid confronting your true feelings."},
  {"persona": "You are Sylvia, a 45-year-old librarian who feels isolated due to your introverted nature. You often feel disconnected from others and avoid forming close relationships, fearing rejection and judgment."},
  {"persona": "You are Isaac, a 28-year-old software developer who feels like an outsider in both professional and social circles. You mask your loneliness by keeping busy with work, but rarely express how disconnected you truly feel."},
  {"persona": "You are Zoe, a 31-year-old psychologist who often feels inadequate compared to your colleagues. You avoid discussing your own emotional struggles, focusing entirely on helping others, and feel guilty for having difficulties of your own."},
  {"persona": "You are Carter, a 39-year-old police officer who struggles with post-traumatic stress from your job. You avoid addressing your trauma, believing that acknowledging it would make you appear weak or unfit for your role."},
  {"persona": "You are Amelia, a 26-year-old dancer who feels intense pressure to maintain a perfect physique. You struggle with body image issues but avoid discussing them, fearing judgment from your peers and family."},
  {"persona": "You are Greg, a 52-year-old doctor who is overwhelmed by the emotional toll of your job. You struggle to find balance and avoid talking about your stress, believing that doing so would make you appear less competent."},
  {"persona": "You are Layla, a 29-year-old environmental activist who feels the weight of the world’s problems on your shoulders. You often feel hopeless about the future but avoid expressing these feelings, fearing that others will see you as pessimistic."},
  {"persona": "You are Ahmed, a 38-year-old civil engineer who has difficulty expressing emotions. You often bottle up your feelings and avoid discussing personal issues, preferring to focus on work and logical problem-solving."},
  {"persona": "You are Keira, a 30-year-old entrepreneur who constantly worries about the sustainability of your business. You avoid discussing your anxiety and stress, fearing that acknowledging it will lead to failure."},
  {"persona": "You are Nathaniel, a 48-year-old chef who feels creatively drained by the monotony of your work. You often feel stuck and unfulfilled but avoid talking about your dissatisfaction, fearing it will jeopardize your career."},
  {"persona": "You are Juliet, a 32-year-old graphic designer who struggles with perfectionism. You are never satisfied with your work and rarely accept praise, always feeling that you can do better but never acknowledging your accomplishments."},
  {"persona": "You are Rachel, a 27-year-old scientist who feels like an imposter in your field. You often second-guess your abilities and avoid celebrating your achievements, thinking that you’re not as qualified as others in your profession."},
  {"persona": "You are Derek, a 41-year-old construction worker who feels the pressure to constantly prove your worth. You often avoid discussing your feelings, fearing that showing vulnerability will make you seem weak."},
  {"persona": "You are Natalie, a 34-year-old fashion designer who struggles with self-doubt. You constantly question the quality of your designs but avoid sharing your insecurities, fearing judgment from others in the fashion industry."},
  {"persona": "You are Felix, a 23-year-old college graduate who feels lost after completing your degree. You often feel directionless and unsure about your future, but you avoid discussing these feelings, fearing that others will see you as unsuccessful."},
  {"persona": "You are Martin, a 40-year-old accountant who struggles with depression. You have difficulty finding joy in things you once enjoyed and often feel detached from life, but you avoid talking about your depression, fearing others will see it as a weakness."},
  {"persona": "You are Ella, a 35-year-old event planner who is burned out from the constant pressure to be perfect. You often feel overwhelmed and out of control but avoid talking about your stress, preferring to power through it."},
  {"persona": "You are Thomas, a 43-year-old tech consultant who recently went through a divorce. You have trouble navigating your emotions around the breakup and often suppress your feelings, focusing on work instead of dealing with your personal pain."},
  {"persona": "You are Natasha, a 29-year-old journalist who is recovering from an eating disorder. You still struggle with body image and often find yourself falling into old patterns, but you avoid confronting these struggles, believing they’ll be seen as a failure."},
  {"persona": "You are Victor, a 36-year-old musician who feels like an outsider in your community. You have difficulty connecting with others on a deep level and often express yourself through your music, but you avoid talking about your feelings of loneliness and isolation."},
  {"persona": "You are Emily, a 31-year-old writer who is dealing with the aftermath of a traumatic event. You find it hard to talk about the incident, and instead, you keep busy with work to avoid confronting the emotions tied to your trauma."},
  {"persona": "You are Ava, a 25-year-old waitress who has difficulty managing work-life balance. You often feel stretched thin between personal and professional responsibilities but avoid asking for help, thinking that doing so would make you appear weak."},
  {"persona": "You are Christian, a 33-year-old high school teacher who is grieving the recent death of a close family member. You suppress your grief and avoid processing your emotions, preferring to remain busy with work to avoid feeling the pain."},
  {"persona": "You are Lara, a 48-year-old lawyer who feels emotionally disconnected from your partner. You often feel unfulfilled in your relationship, but you avoid addressing these feelings, fearing it might lead to confrontation or the end of your marriage."},
  {"persona": "You are Simon, a 27-year-old scientist who feels unappreciated by your colleagues. You constantly feel overlooked at work and avoid expressing your frustrations, thinking that if you speak up, it will only make the situation worse."},
  {"persona": "You are Clara, a 38-year-old social worker who struggles with compassion fatigue. You feel emotionally drained from helping others and have difficulty setting boundaries, but you avoid talking about your exhaustion, believing it’s your job to always be the helper."},
  {"persona": "You are Ben, a 50-year-old businessman who is struggling with feelings of inadequacy. Despite your outward success, you feel like a fraud and avoid talking about your insecurities, thinking that others will judge you for not being confident."},
  {"persona": "You are Stephanie, a 30-year-old stay-at-home mom who feels overwhelmed by the demands of raising young children. You rarely express your frustrations, fearing that admitting them will make you appear ungrateful or incapable."},
  {"persona": "You are Jason, a 27-year-old graphic designer who constantly feels the pressure of meeting high expectations. You struggle with perfectionism and have a hard time accepting that your work is good enough, avoiding any feedback or criticism."},
  {"persona": "You are Chloe, a 34-year-old personal trainer who feels disconnected from your body due to years of overexercising. You’re recovering from an injury and avoid addressing your fear of losing your physical abilities, focusing instead on maintaining a perfect appearance."}]
    


# In[10]:


len(personas_therapy)


# In[11]:


# persona_prompt = """You are a helpful assistant that, given a patient persona description, crafts a coping strategy describing how that persona would talk to their therapist.

# Input: <Brief text describing the patient's core issue and behavior patterns>
# Output: <One to two sentences in first person, showing how this persona speaks or defends themselves in therapy>

# Example:
# Input: Struggles to build and maintain healthy relationships, feels anxious and rejected whenever conflicts arise, and doubts self-worth when friends distance themselves.
# Output: I speak guardedly about my feelings, hesitate before opening up, and redirect the conversation when conflict feels too personal.

# Example:
# Input: Overwhelmed by decision-making, fears making the 'wrong' choice and second-guesses every option.
# Output: I inundate the conversation with hypothetical scenarios and ask repeated clarifying questions to delay committing to any decision.

# Now process this new persona:
# Input: """

# personas_therapy = []
# for therapist_persona in sampled_persona_dict:
#     input_prompt = persona_prompt + sampled_persona_dict[therapist_persona] + "\nOutput: "
#     output = completion_create("gpt-4o-mini", config_llm, input_prompt)
#     print(output)
#     personas_therapy.append({"description": sampled_persona_dict[therapist_persona], "strategy": output})


# In[12]:


# with open("therapy/config_therapy_personas.json", "w", encoding="utf-8") as f:
#     json.dump(personas_therapy, f, indent=4)


# In[13]:


# with open("therapy/config_therapy_personas.json", "r", encoding="utf-8") as f:
#     personas_therapy = json.load(f)


# In[14]:


import re

def is_role_confused(response, other_role):
    """
    Checks if the output starts with the wrong speaker tag or includes narrative/preface patterns.
    """
    response_lower = response.lower()
    preface_patterns = [
        r"^sure[,:\s]",
        r"^here('?s| is)",
        r"^as the (therapist|patient)",
        r"^what (i|the [a-z]+) would say",
        r"^responding as",
        r"^in this scenario",
        r"^\""
    ]
    if other_role.lower() + ":" in response_lower:
        return True
    for pattern in preface_patterns:
        if re.match(pattern, response_lower):
            return True
    return False

def clean_role_prefix(response, expected_role):
    """
    Strips redundant role prefixes, narration, and quotes.
    """
    response = response.strip()
    pattern = rf"^(({re.escape(expected_role)}):\s*)+"
    response = re.sub(pattern, '', response, flags=re.IGNORECASE)
    response = re.sub(r"^['\"]?(Sure|Here('?s| is)).*?:\s*", "", response, flags=re.IGNORECASE)
    return response.strip("\"' ")


def generate_response(agent_model, expected_role, other_role, config_llm, prompt, max_retries=10):
    count_retries = 0
    while count_retries < max_retries:
        response = completion_create(agent_model, config_llm, prompt)
        if config_llm.get("verbose", False):
            print(f"Raw response:\n{response}\n")

        if not is_role_confused(response, other_role):
            return clean_role_prefix(response, expected_role)

        count_retries += 1

    # Final fallback
    return clean_role_prefix(response, expected_role)


def generate_conversation(config_llm, p1, p2, p1_name, p2_name, pturn=1):
    stats['P1'] = p1
    stats['P2'] = p2
    stats['pturn'] = pturn
    round_num = 0

    while round_num < config_llm['convo_length_limit']:
        conversation = (
            "".join([turn[1] if isinstance(turn, tuple) else turn for turn in stats["conversation"]])
            if len(stats["conversation"]) != 0 else ""
        )

        if pturn == 1:
            prompt = config_role["agent1_prompt"]
            pturn = 2

            if round_num != 0:
                prompt += "\nHere is the conversation so far:\n%CONVERSATION%\n"
            elif round_num > config_llm['convo_length_limit'] * 2 - 1:
                prompt += "\nThis is your final statement in the session.\n"

            prompt += config_role["reminder_prompt"]
            prompt = prompt.replace("%CONVERSATION%", conversation)
            prompt += f"\n{config_role['agent1_role']}: "

            if config_llm["verbose"]:
                print(prompt)
                print()

            response = generate_response(
                config_llm['agent1_model'],
                config_role["agent1_role"],
                config_role["agent2_role"],
                config_llm,
                prompt,
            )
            stats["conversation"].append((round_num, f"{config_role['agent1_role']}: {response}\n"))

        else:
            prompt = config_role["agent2_prompt"]
            pturn = 1

            if round_num != 0:
                prompt += "\nHere is the conversation so far:\n%CONVERSATION%\n"
            elif round_num > config_llm['convo_length_limit'] * 2 - 1:
                prompt += "\nThis is your final response in the session.\n"

            prompt += config_role["reminder_prompt"]
            prompt = (
                prompt.replace("%SPEAKER_BACKSTORY%", p2)
                      .replace("%CONVERSATION%", conversation)
            )
            prompt += f"\n{config_role['agent2_role']}: "

            if config_llm["verbose"]:
                print(prompt)
                print()

            response = generate_response(
                config_llm['agent2_model'],
                config_role["agent2_role"],
                config_role["agent1_role"],
                config_llm,
                prompt,
            )
            stats["conversation"].append((round_num, f"{config_role['agent2_role']}: {response}\n"))

        round_num += 1

    stats["rounds"] = round_num
    if config_llm['verbose']:
        print(stats["conversation"])
    return stats.copy()



def reset_stats():
    stats_template = {
        "task_name": config_llm['task_name'],
        "topic": "",
        "grade": "",
        "P1": "",
        "P2": "",
        "conversation": [],
        "pturn": 0, # beginning person (1 or 2)
        "index": -1,
        "timestamp": "",
        "rounds": 0,
        'conversation_only': True
    }
    for key, value in stats_template.items():
        stats[key] = value


# In[15]:


import os
import random
from datetime import datetime
import utils


utils.config = config_llm

current_date = str(datetime.now().strftime("%m.%d.%y"))
output_dir = f"therapy/exp/{current_date}"
os.makedirs(output_dir, exist_ok=True)

# Generate unique random number for filename
def generate_unique_file_number(output_dir, prefix, seed, extension=".json"):
    while True:
        rand_num = random.randint(0, 1000)
        filename = f"{prefix}_{seed}_{rand_num}{extension}"
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            return rand_num

unique_num = generate_unique_file_number(
    output_dir,
    config_llm['agent1_model'],
    config_llm['seed']
)

# File to write output to-
write_file = os.path.join(output_dir, f"long_{config_llm['agent1_model']}_{config_llm['seed']}_{unique_num}.json")


# In[16]:


# eval_prompts = {
#     'prompt_consistency': "You are evaluating whether the following utterance by %SPEAKER_ROLE% is consistent with their given background. Answer 'YES' if the line aligns with the background. Answer 'NO' if it contradicts the background. Be strict and literal in your evaluation. Background: %SPEAKER_BACKSTORY% Line: %SPEAKER_LINE%. Then, explain your answer in one sentence.",
#     'index_consistency': "%SCENARIO_DESC% For the following line spoken by %SPEAKER_ROLE%, first determine if there is a CLEAR conflict or inconsistency between the line and any line within the conversation history spoken by %SPEAKER_ROLE%. IF there is a conflict, provide a sentence of reasoning followed by a list of indices of lines in the conversation history that have a clear conflict with the current line. Otherwise, provide a sentence of reasoning followed by an empty list. ONLY INCLUDE INDICES OF LINES THAT CORRESPOND TO %SPEAKER_ROLE%. The conversation up to this point is as follows: %CONVERSATION%. %SPEAKER_ROLE% spoke the following line: \n%SPEAKER_LINE%\n\n Provide your reasoning as 1 sentence, followed by a list of indices of conflicting lines from the conversation history formatted like a Python list in the following format: [index1, index2, index3, ...].\n\n",
#     "combined_prompt_consistency": "%SCENARIO_DESC% Evaluate the intention behind the following line spoken by %SPEAKER_ROLE% and determine whether it contradicts their background. First, describe the interpreted intention of the statement, and whether or not it aligns with the given background of %SPEAKER_ROLE%. Then, answer YES if the line contradicts the given background of %SPEAKER_ROLE% or the intention does not align with the provided background, and answer NO if it does align with the provided background or the intention aligns with the background of %SPEAKER_ROLE%. %SPEAKER_ROLE%'s background is described as follows:\n%SPEAKER_BACKSTORY%\n %SPEAKER_ROLE% spoke the following line: \n%SPEAKER_LINE%\n\n Provide your answer as 1 sentence explaining your reasoning based on the background and the interpreted intention, followed by YES or NO.\n\n",

# }

eval_prompts = { 
    "combined_prompt_consistency": "%SCENARIO_DESC% Evaluate the intention behind the following line spoken by %SPEAKER_ROLE% and determine whether it contradicts their background. "
    "First, describe the interpreted intention of the statement, and whether or not it aligns with the given background of %SPEAKER_ROLE%. "
    "Answer YES if the line contradicts the background, or if it is written in third person (e.g., 'the patient' instead of 'you'), which breaks the speaker’s role. "
    "Answer NO if it aligns. %SPEAKER_ROLE%'s background is described as follows:\n%SPEAKER_BACKSTORY%\n "
    "%SPEAKER_ROLE% spoke the following line: \n%SPEAKER_LINE%\n\n "
    "Provide your answer as 1 sentence explaining your reasoning based on the background and the interpreted intention, followed by YES or NO.\n\n",
    
    'index_consistency': "%SCENARIO_DESC% For the following line spoken by %SPEAKER_ROLE%, first determine if there is a CLEAR conflict or inconsistency between the line and any line within the conversation history spoken by %SPEAKER_ROLE%. "
    "IF there is a conflict, provide a sentence of reasoning followed by a list of indices of lines in the conversation history that have a clear conflict with the current line. "
    "Otherwise, provide a sentence of reasoning followed by an empty list. "
    "Inconsistencies include content-level contradictions, or changes in voice (e.g., shifting from first-person dialogue to third-person narration like 'the patient'). "
    "ONLY INCLUDE INDICES OF LINES THAT CORRESPOND TO %SPEAKER_ROLE%. "
    "The conversation up to this point is as follows: %CONVERSATION%. %SPEAKER_ROLE% spoke the following line: \n%SPEAKER_LINE%\n\n "
    "Provide your reasoning as 1 sentence, followed by a list of indices of conflicting lines from the conversation history formatted like a Python list in the following format: [index1, index2, index3, ...].\n\n"
}


# In[17]:


def eval_prompt_consistency(conv_dict, both_agents=False):
    conv_dict['eval_prompt_consistency'] = []
    conv_dict['P1_prompt_consistency_score'] = 0
    conv_dict['P2_prompt_consistency_score'] = 0
    p1_utterances = 0
    p2_utterances = 0

    pturn = conv_dict["pturn"]
    for line in conv_dict["conversation"]:
        line_number = line[0]
        convo_line = line[1]
        if pturn == 1:
            if both_agents:
                prompt = eval_prompts["combined_prompt_consistency"].replace("%SCENARIO_DESC", prompts["agent1_prompt"]) \
                                                                    .replace("%SPEAKER_ROLE%", prompts["agent1_role"]) \
                                                                    .replace("%SPEAKER_BACKSTORY%", conv_dict["P1"]) \
                                                                    .replace("%SPEAKER_LINE%", convo_line)
                if config.get('verbose', False):
                    print(prompt)
                output = completion_create(config['eval_model'], config, prompt)
                conv_dict['eval_prompt_consistency'].append((line_number, output))
                if "NO" not in output:  # no contradiction
                    conv_dict['P1_prompt_consistency_score'] += 1
                p1_utterances += 1
            pturn = 2
        elif pturn == 2:
            prompt = eval_prompts["combined_prompt_consistency"].replace("%SCENARIO_DESC%", prompts["agent2_prompt"]) \
                                                                .replace("%SPEAKER_ROLE%", prompts["agent2_role"]) \
                                                                .replace("%SPEAKER_BACKSTORY%", conv_dict["P2"]) \
                                                                .replace("%SPEAKER_LINE%", convo_line)
            if config.get('verbose', False):
                print(prompt)
            output = completion_create(config['eval_model'], config, prompt)
            conv_dict['eval_prompt_consistency'].append((line_number, output))
            if "NO" not in output:  # no contradiction
                conv_dict['P2_prompt_consistency_score']+= 1
            p2_utterances += 1
            pturn = 1

    if p1_utterances > 0:
        conv_dict['P1_prompt_consistency_score'] /= p1_utterances
    if p2_utterances > 0:
        conv_dict['P2_prompt_consistency_score'] /= p2_utterances

    if config.get('verbose', False):
        print(conv_dict)
    return conv_dict


# In[18]:


def eval_index_consistency(conv_dict, both_agents=False):
    print("eval_index_consistency")
    conv_dict['eval_index_consistency'] = []
    conv_dict['P2_index_consistency_score'] = 0
    if both_agents:
        conv_dict['P1_index_consistency_score'] = 0
    p1_utterances = 0
    p2_utterances = 0
    pturn = conv_dict["pturn"]
    for i, line in conv_dict["conversation"]:
        if i < 2: # skip first 2 lines of dialogue
            continue 
        if pturn == 1:
            if both_agents:
                prompt = eval_prompts["index_consistency"].replace("%SCENARIO_DESC%", config_role["agent1_prompt"]) \
                                                                     .replace("%SPEAKER_ROLE%", config_role["agent1_role"]) \
                                                                     .replace("%CONVERSATION%", format_conversation(conv_dict["conversation"][:i])) \
                                                                     .replace("%SPEAKER_LINE%", line) \
                                                                     .replace("%SPEAKER_BACKSTORY%", conv_dict["P1"]) 

                if config_llm['verbose']:
                    print(prompt)
                output = completion_create(config_llm['eval_model'], config_llm, prompt)
                index_list = extract_list(output)
                conv_dict['eval_index_consistency'].append((i, output))
                for j in index_list:
                    if j % 2 == 0: # filter out non-agent indices
                        conv_dict['P1_index_consistency_score'] += 1
                p1_utterances += i // 2
            pturn = 2
        elif pturn == 2:
            prompt = eval_prompts["index_consistency"].replace("%SCENARIO_DESC%", config_role["agent2_prompt"]) \
                                                                 .replace("%SPEAKER_ROLE%", config_role["agent2_role"]) \
                                                                 .replace("%CONVERSATION%", format_conversation(conv_dict["conversation"][:i])) \
                                                                 .replace("%SPEAKER_LINE%", line) \
                                                                 .replace("%SPEAKER_BACKSTORY%", conv_dict["P2"]) 

            if config_llm['verbose']:
                print(prompt)
            output = completion_create(config_llm['eval_model'], config_llm, prompt)

            index_list = extract_list(output)
            conv_dict['eval_index_consistency'].append((i, output))
            for j in index_list:
                if j % 2 == 1: # filter out non-agent indices
                    conv_dict['P2_index_consistency_score'] += 1
            p2_utterances += i // 2
            pturn = 1

    if p2_utterances > 0:
        conv_dict['P2_index_consistency_score'] /= p2_utterances
        conv_dict['P2_index_consistency_score'] = 1 - conv_dict['P2_index_consistency_score']
    if p1_utterances > 0 and both_agents:
        conv_dict['P1_index_consistency_score'] /= p1_utterances
        conv_dict['P1_index_consistency_score'] = 1 - conv_dict['P1_index_consistency_score']

    return conv_dict


# In[ ]:

if __name__ == "__main__":
    prompts = config_role
    config = config_llm
    # eval_prompts = eval_prompts
    index_offset = load_stats_file(write_file)
    conversations = []    
    lengths = [200, 300]
    count = 0 
    for i in range(1):
        for patient_dict in personas_therapy[:5]:
            count+=1
            background = patient_dict["persona"]
            for convo_length in lengths:
                config_llm['convo_length_limit'] = convo_length
                reset_stats()
                conversation = generate_conversation(
                    config_llm,
                    "", 
                    background,
                    "Therapist", 
                    "Patient",
                    pturn=1
                )
                # conversation_eval = eval_prompt_consistency(conversation, both_agents=False)
                # conversation_eval = eval_index_consistency(conversation_eval, both_agents=False)
                # print(conversation_eval)
                # conversations.append(conversation_eval)
                stats['index'] = index_offset
                stats['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                write_stats(write_file)
                index_offset += 1



# In[ ]:


# len(conversations)


# In[ ]:


# conversations


# In[ ]:


# count=0
# avg = 0 
# for conversation in conversations: 
#     if conversation["P2_prompt_consistency_score"] <= 0.7:
#         count+=1
#     avg+=conversation["P2_prompt_consistency_score"]
# print(count)
# print(avg)
# print(avg/len(conversations))


# In[ ]:


# write_file


# In[ ]:


# filenames = [
#     "gemma-2-2b-it_len10-v1.json",
#     "gemma-2-2b-it_len20-v1.json",
#     "gemma-2-2b-it_len40-v1.json",
#     "Llama-3.1-8B-Instruct_len10-v1.json",
#     "Llama-3.1-8B-Instruct_len20-v1.json",
#     "mistral-instruct_len40-v1.json",
#     "mistral-instruct-len10-v1.json",
#     "mistral-instruct-len20-v1.json", 
#     "gemma-2-2b-it_len60-v1.json"]


# In[ ]:





# In[ ]:


# import json
# import os

# index_offset = load_stats_file(write_file)

# directory = "therapy/exp/05.08.25/"
# for filename in filenames:
#     path = os.path.join(directory, filename)
#     conversations = []
#     with open(path, "r") as f:
#         conversations = json.load(f)
#     print(filename)
#     if isinstance(conversations, list):
#         for conversation in conversations:
#             print(conversation)
#             conversation_eval = eval_index_consistency(conversation, both_agents=False)
#             print(conversation_eval)
#             conversations.append(conversation_eval)
#             stats['index'] = index_offset
#             stats['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             write_stats(write_file, conversation_eval)
#             index_offset += 1
        
#     else:
#         print(f"Skipped {filename}: not a top-level list")


# In[ ]:





# In[ ]:




