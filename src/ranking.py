from sklearn.metrics.pairwise import cosine_similarity

def rank_resumes(job_embedding, resume_embeddings, experience_years):
    similarity = cosine_similarity([job_embedding], resume_embeddings)[0]
    exp_score = experience_years / experience_years.max()

    final_score = 0.6 * similarity + 0.4 * exp_score
    return final_score
