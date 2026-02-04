from ingest import load_data
from preprocess import clean_text
from embeddings import embed_texts
from ranking import rank_resumes

df = load_data("data/synthetic_resumes.csv")

df["resume_text"] = (
    df["Skills"].astype(str) + " " +
    df["Certifications"].astype(str) + " " +
    df["Current_Job_Title"].astype(str)
)

df["resume_text"] = df["resume_text"].apply(clean_text)

job_desc = clean_text(df["Target_Job_Description"].iloc[0])

resume_embeddings = embed_texts(df["resume_text"].tolist())
job_embedding = embed_texts([job_desc])[0]

df["final_score"] = rank_resumes(
    job_embedding,
    resume_embeddings,
    df["Experience_Years"]
)

top_candidates = df.sort_values("final_score", ascending=False).head(10)
print(top_candidates[["Name", "Current_Job_Title", "final_score"]])
