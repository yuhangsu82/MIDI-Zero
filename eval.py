import sys
import os
from utils import get_midi_files, calculate_metrics
from data.dataset import MidiEmbeddingDataset, preprocess_and_save_midi_data
import numpy as np
from tqdm import tqdm
from models.MidiResNet import MidiResNet
from models.MidiTransNet import MidiTransNet
import torch
import faiss
import torch.nn.functional as F
import pandas


def generate_feature(
    data_source, output_root_dir, feature_dim, device, batch_size, model, pitch_only
):
    """
    Generate features for the given data source.

    Args:
        data_source: Data source for feature extraction.
        output_root_dir: Root directory for saving the feature.
        feature_dim: Dimension of the extracted feature.
        device: Device for model inference.
        batch_size: Batch size for feature extraction.
        model: Model for feature extraction.
        pitch_only: Use pitch-only model if True.

    Returns:
        arr: Extracted features.
    """

    model.eval()
    db_nb = len(data_source)
    pbar = tqdm(data_source, total=db_nb, disable=False, desc='generate feature')
    arr_shape = (len(data_source.dataset), feature_dim)
    # print(arr_shape)
    arr = np.memmap(
        str(output_root_dir) + '.mm', dtype='float32', mode='w+', shape=arr_shape
    )
    np.save(str(output_root_dir) + '_shape.npy', arr_shape)
    for i, data in enumerate(pbar):
        # emb = model(anchor.to(device)).detach().cpu()
        if pitch_only:
            anchor = data
            emb = F.normalize(model(anchor.to(device)), dim=-1).detach().cpu()
        else:
            anchor, time_info = data
            emb = F.normalize(model(anchor.to(device), time_info.to(device)), dim=-1).detach().cpu()
        arr[i * batch_size : (i + 1) * batch_size, :] = emb.numpy()
    arr.flush()
    return arr


def batch_search_faiss(embeddings, index, gt_file_path, batch_size=1024, top_k=10):
    """
    Perform batch search on a Faiss index.

    Args:
        embeddings (numpy.ndarray): Query embeddings, shape (num_queries, dim).
        index (faiss.Index): Faiss index for retrieval.
        gt_file_path (str): File path to ground truth file.
        batch_size (int): Number of queries to process per batch.
        top_k (int): Number of nearest neighbors to retrieve for each query.
        output_file (str): File path to save the search results.
    
    Returns:
        results (Recall@1, Recall@k, mrr, nDCG)
    """
    num_queries = embeddings.shape[0]
    results = np.zeros((num_queries, top_k), dtype=np.int64)  # Store retrieved IDs
    dists = np.zeros((num_queries, top_k), dtype=np.float32)  # Store distances
    gt = pandas.read_csv(gt_file_path)
    gt_dict = {}
    for i, query_name in enumerate(gt["query"]):
        gt_dict[query_name] = gt["gt"][i]

    for start_idx in tqdm(range(0, num_queries, batch_size), desc='search'):
        end_idx = min(start_idx + batch_size, num_queries)
        batch_embeddings = embeddings[start_idx:end_idx]
        D, I = index.search(batch_embeddings, top_k)
        results[start_idx:end_idx] = I
        dists[start_idx:end_idx] = D
    
    result_dict = {}
    for i in tqdm(range(num_queries), desc='calculate metrics'):
        query_name, _ = query_dataset.get_file_name(i)

        if query_name not in result_dict:
            result_dict[query_name] = dict()

        for j in range(top_k):
            result_name, _ = db_dataset.get_file_name(results[i][j])
            result_name = result_name

            if result_name in result_dict[query_name]:
                result_dict[query_name][result_name] += dists[i][j]
            else:
                result_dict[query_name][result_name] = dists[i][j]

    num_queries = len(result_dict)
    gt = []
    retrieval_results = []

    for key, value in result_dict.items():
        result = sorted(value.items(), key=lambda x: x[1], reverse=True)
        result_ids = [x[0] for x in result][:top_k]
        result_dist = [x[1] for x in result][:top_k]
        query_label = gt_dict[key]
        gt.append(query_label)
        retrieval_results.append(result_ids)

    recall_at_1, recall_at_10, mrr, nDCG = calculate_metrics(retrieval_results, gt, top_k=10)

    return recall_at_1, recall_at_10, mrr, nDCG


def eval(
    db_path, query_path, gt_file_path, checkpoint_path, batch_size=256, feature_dim=128, note_max=12, pitch_only=True
):
    """
    Evaluate the model on the given dataset.

    Args:
        db_path: Path to the database.
        query_path: Path to the query set.
        gt_file_path: Path to the ground truth file.
        checkpoint_path: Path to the model checkpoint.
        batch_size: Batch size for feature extraction.
        feature_dim: Dimension of the extracted feature.
        note_max: Maximum number of notes to consider.
        pitch_only: Use pitch-only model if True.
    
    Returns:
        recall_at_1, recall_at_10, mrr, nDCG
    """
    db_files = get_midi_files(db_path)
    query_files = get_midi_files(query_path)

    db_json_path = os.path.join(os.path.dirname(db_path), 'db.json')
    query_json_path = os.path.join(os.path.dirname(query_path), 'query.json')

    preprocess_and_save_midi_data(db_files, db_json_path, pitch_only, source_type='performance', time_resolution=0.1)
    preprocess_and_save_midi_data(query_files, query_json_path, pitch_only, source_type='performance', time_resolution=0.1)

    global db_dataset, query_dataset
    db_dataset = MidiEmbeddingDataset(data_path=db_json_path, mode='eval', note_max=note_max, step_size=2, pitch_only=pitch_only, data_format='matrix')
    db_loader = torch.utils.data.DataLoader(db_dataset, batch_size, shuffle=False)
    query_dataset = MidiEmbeddingDataset(data_path=query_json_path, mode='eval', note_max=note_max, step_size=2, pitch_only=pitch_only, data_format='matrix')
    query_loader = torch.utils.data.DataLoader(query_dataset, batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MidiResNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda:0', weights_only=True)['model'])

    # db_shape = np.load(db_path + '_shape.npy')
    # db = np.memmap(
    #     db_path + '.mm',
    #     dtype='float32',
    #     mode='r',
    #     shape=(db_shape[0], feature_dim),
    # )
    # query_shape = np.load(query_path + '_shape.npy')
    # query = np.memmap(
    #     query_path + '.mm',
    #     dtype='float32',
    #     mode='r',
    #     shape=(query_shape[0], feature_dim),
    # )

    db = generate_feature(db_loader, db_path, 128, device, batch_size, model, pitch_only)
    query = generate_feature(query_loader, query_path, 128, device, batch_size, model, pitch_only)
    index = faiss.IndexFlatIP(feature_dim)
    index.add(db)
    recall_at_1, recall_at_10, mrr, nDCG = batch_search_faiss(query, index, gt_file_path)

    return recall_at_1, recall_at_10, mrr, nDCG


if __name__ == "__main__":
    db_path = './runs/database/beginnerPiano/db'
    query_path = './runs/database/beginnerPiano/query'
    gt_file_path = './runs/database/beginnerPiano/query_gt.csv'
    checkpoint_path = './runs/checkpoint/beginerPiano.pth'

    # db_path = './runs/database/guitarset/db'
    # query_path = './runs/database/guitarset/query'
    # gt_file_path = './runs/database/guitarset/query_gt.csv'
    # checkpoint_path = './runs/checkpoint/guitarset.pth'

    # db_path = './runs/database/m4singer/db'
    # query_path = './runs/database/m4singer/query'
    # gt_file_path = './runs/database/m4singer/query_gt.csv'
    # checkpoint_path = './runs/checkpoint/m4singer.pth'

    eval(db_path, query_path, gt_file_path, checkpoint_path, note_max=12)

    # for id in range(0, 31):
    #     checkpoint_path = f'./checkpoint/test/mem_{id}.pth'
    #     # recall_at_1, recall_at_10, mrr, nDCG = eval(db_path, query_path, checkpoint_path, level='file')
    #     recall_at_1, recall_at_10, mrr, nDCG = eval(db_path, query_path, checkpoint_path, level='file', note_max=6)

    #     with open(os.path.dirname(checkpoint_path) + '/results.txt', 'a') as f:
    #         f.write(f"Model {id}: Recall@1: {recall_at_1}, Recall@10: {recall_at_10}, mrr: {mrr}, nDCG: {nDCG}\n")
