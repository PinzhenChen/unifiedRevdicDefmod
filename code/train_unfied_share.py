import sys
import argparse
import itertools
import json
import logging
import pathlib
import pprint
import random
import skopt
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import data
import model_baseline as models
from model_unified import OurModel


def _set_random_seeds(random_seed=2022):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


# _set_random_seeds(int(time.time()))  # for ensembling
_set_random_seeds(2022) # for reproducibility

logger = logging.getLogger(pathlib.Path(__file__).name)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(handler)


def get_parser(
        parser=argparse.ArgumentParser(
            description="A trainer for the autoencoder model, involving both task at the same time."
        ),
):
    parser.add_argument(
        "--do_htune",
        action="store_true",
        help="whether to perform hyperparameter tuning",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="whether to train a model from scratch"
    )
    parser.add_argument(
        "--do_pred", action="store_true", help="whether to produce predictions"
    )
    parser.add_argument(
        "--pred_direction",
        type=str,
        choices=("definition", "embedding"),
        help="generate word embeddings or definitions"
    )
    parser.add_argument(
        "--train_file", type=pathlib.Path, help="path to the train file"
    )
    parser.add_argument("--dev_file", type=pathlib.Path, help="path to the dev file")
    parser.add_argument("--test_file", type=pathlib.Path, help="path to the test file")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--word_emb_dim", type=int, default=256,
                        help="Word embedding dimension size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1.0e-4, help="Maximum learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1")
    parser.add_argument("--beta2", type=float, default=0.99, help="beta2")
    parser.add_argument("--weight_decay", type=float, default=1.0e-6, help="weight decay")
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cpu"),
        help="devive to be used, \'cpu\' or \'cuda:$id\'",
    )
    parser.add_argument(
        "--embedding_arch",
        type=str,
        choices=("sgns", "char", "electra", "bert"),
        help="embedding architecture to use",
    )
    parser.add_argument(
        "--summary_logdir",
        type=pathlib.Path,
        help="where to write logs for future analysis",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        help="where to save model and vocab",
    )
    parser.add_argument(
        "--pred_file",
        type=pathlib.Path,
        help="where to save predictions",
    )
    parser.add_argument(
        "--spm_model_path",
        type=pathlib.Path,
        default=None,
        help="use sentencepiece model, if required train and save it here",
    )
    parser.add_argument(
        "--embedding_loss",
        type=str,
        default="mse",
        choices=("mse", "cosine"),
        help="what loss to use for embeddings",
    )
    parser.add_argument(
        "--token_loss",
        type=str,
        default="ce",
        choices=("ce"),
        help="what loss to use for tokens",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="label smoothing")
    parser.add_argument(
        "--reconstruction",
        action="store_true",
        help="whether to train the model with a reconstruction task",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="whether to have a skip connection on the autoencoder",
    )

    return parser


def get_search_space():  # TODO add number of layers
    """get hyperparameters to optimize for"""
    search_space = [
        skopt.space.Real(1e-7, 1e-2, "log-uniform", name="learning_rate"),
        skopt.space.Real(0.0, 1.0, "uniform", name="weight_decay"),
        skopt.space.Real(0.9, 1.0 - 1e-8, "log-uniform", name="beta_a"),
        skopt.space.Real(0.9, 1.0 - 1e-8, "log-uniform", name="beta_b"),
        skopt.space.Real(0.0, 0.9, "uniform", name="dropout"),
        skopt.space.Real(0.0, 1.0, "uniform", name="warmup_len"),
        skopt.space.Real(0.0, 1.0 - 1e-8, "uniform", name="label_smoothing"),
        skopt.space.Integer(1, 100, "log-uniform", name="batch_accum"),
        skopt.space.Integer(1, 4, "uniform", name="n_head_pow"),
        skopt.space.Integer(2, 6, "uniform", name="n_layers"),
    ]
    return search_space


def train(
        train_file,
        dev_file,
        embedding_arch,
        summary_logdir,
        save_dir,
        device,
        batch_size,
        embedding_loss,
        token_loss,
        label_smoothing,
        skip_flag,
        recon_flag,
        spm_model_path=None,
        epochs=100,
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.999,
        weight_decay=1e-6,
        patience=5,
        batch_accum=1,
        dropout=0.3,
        warmup_len=0.1,
        n_head=4,
        n_layers=4,
):
    assert train_file is not None, "Missing dataset for training"
    assert dev_file is not None, "Missing dataset for development"

    # 1. get data, vocabulary, summary writer
    logger.debug("Preloading data")
    save_dir.mkdir(parents=True, exist_ok=True)

    ## make datasets
    train_dataset = data.get_train_dataset(train_file, spm_model_path, save_dir)
    dev_dataset = data.get_dev_dataset(dev_file, spm_model_path, save_dir, train_dataset)

    ## assert they correspond to the task
    assert train_dataset.has_gloss, "Training dataset contains no gloss."
    if embedding_arch == "electra":
        assert train_dataset.has_electra, "Training datatset contains no vector."
    else:
        assert train_dataset.has_vecs, "Training datatset contains no vector."
    assert dev_dataset.has_gloss, "Development dataset contains no gloss."
    if embedding_arch == "electra":
        assert dev_dataset.has_electra, "Development dataset contains no vector."
    else:
        assert dev_dataset.has_vecs, "Development dataset contains no vector."
    ## make dataloader
    train_dataloader = data.get_dataloader(train_dataset, batch_size=batch_size)
    dev_dataloader = data.get_dataloader(dev_dataset, batch_size=batch_size, shuffle=False)
    ## make summary writer
    summary_writer = SummaryWriter(summary_logdir)
    train_step = itertools.count()  # to keep track of the training steps for logging

    # 2. construct model
    ## Hyperparams
    logger.debug("Setting up training environment")
    model = OurModel(
        vocab=dev_dataset.vocab,
        word_emb_dim=len(train_dataset[0][embedding_arch]),
        skip=skip_flag
    )

    ## Parameter initialization
    for name, param in model.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
        else:  # gain parameters of the layer norm
            nn.init.ones_(param)

    model.ae_to_def.embedding = model.def_to_ae.embedding
    model.ae_to_def.positional_encoding = model.def_to_ae.positional_encoding
    model.ae_to_def.v_proj.weight = model.ae_to_def.embedding.weight

    # check how many model parameters
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def count_total_parameters(model):
        return sum(p.numel() for p in model.parameters())

    c = count_trainable_parameters(model)
    logging.info(f'The model has ' + str(c) + ' trainable parameters')
    print(f'The model has ' + str(c) + ' trainable parameters')
    c = count_total_parameters(model)
    logging.info(f'The model has ' + str(c) + ' parameters in total')
    print(f'The model has ' + str(c) + ' parameters in total')
    # exit()

    ## training mode
    model.to(device)
    model.train()

    # 3. declare optimizer & criterion
    ## Hyperparams
    # EPOCHS, LEARNING_RATE, BETA1, BETA2, WEIGHT_DECAY = 10, 1.0e-4, 0.9, 0.999, 1.0e-6
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )

    ## Losses
    if embedding_loss == "mse":
        embedding_criterion = nn.MSELoss()
    elif embedding_loss == "cosine":
        embedding_criterion = nn.CosineEmbeddingLoss()
    else:
        raise NotImplementedError

    if token_loss == "ce":
        if label_smoothing:
            token_criterion = models.LabelSmoothingCrossEntropy(ignore_index=model.padding_idx,
                                                                epsilon=label_smoothing)
        else:
            token_criterion = nn.CrossEntropyLoss(ignore_index=model.padding_idx)
    else:
        raise NotImplementedError

    # 4. train model
    epochs_range = tqdm.trange(epochs, desc="Epochs")
    total_steps = (len(train_dataloader) * epochs) // batch_accum
    scheduler = models.get_schedule(
        optimizer, round(total_steps * warmup_len), total_steps
    )

    best_loss_embed, best_loss_token = float("inf"), float("inf")
    strikes_embed, strikes_token = 0, 0
    vec_tensor_key = f"{embedding_arch}_tensor"

    ## train loop
    for epoch in epochs_range:
        pbar = tqdm.tqdm(
            desc=f"Train {epoch}", total=len(train_dataset), disable=None, leave=False
        )
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            gls = batch["gloss_tensor"].to(device)
            vec = batch[vec_tensor_key].to(device)

            ae_word, ae_def, embed_recon, def_recon, embed_conv, def_conv = model(vec, gls)

            if recon_flag:
                _, _, _, _, embed_conv_recon, def_conv_recon = model(embed_conv, gls)

            # TODO think about signs and bring the two losses to the same scale?
            loss = 0.0
            if embedding_loss == "mse":
                ae_loss = embedding_criterion(ae_word, ae_def)
                embed_recon_loss = embedding_criterion(embed_recon, vec)
                embed_conv_loss = embedding_criterion(embed_conv, vec)
                if recon_flag:
                    embed_conv_recon_loss = embedding_criterion(embed_conv_recon, vec)
            elif embedding_loss == "cosine":
                cosine_label = torch.ones(batch[vec_tensor_key].shape[0],
                                          device=device,
                                          requires_grad=False)
                ae_loss = embedding_criterion(ae_word, ae_def, cosine_label)
                embed_recon_loss = embedding_criterion(embed_recon, vec, cosine_label)
                embed_conv_loss = embedding_criterion(embed_conv, vec, cosine_label)
                if recon_flag:
                    embed_conv_recon_loss = embedding_criterion(embed_conv_recon, vec, cosine_label)
            else:
                raise NotImplementedError

            if token_loss == "ce":
                def_recon_length = def_recon.size(-1)
                def_recon_loss = token_criterion(def_recon.view(-1, def_recon_length), gls.view(-1))
                def_conv_length = def_conv.size(-1)
                def_conv_loss = token_criterion(def_conv.view(-1, def_conv_length), gls.view(-1))
                if recon_flag:
                    def_conv_recon__length = def_conv.size(-1)
                    def_conv_recon_loss = token_criterion(
                        def_conv_recon.view(-1, def_conv_recon__length), gls.view(-1))
            else:
                raise NotImplementedError

            loss += ae_loss + embed_recon_loss + embed_conv_loss + def_recon_loss + def_conv_loss
            if recon_flag:
                loss += embed_conv_recon_loss + def_conv_recon_loss
            loss.backward()
            step = next(train_step)
            grad_remains = True

            if i % batch_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                grad_remains = False

                # keep track of the train loss for this step
                summary_writer.add_scalar(
                    "autoencoder-train/autoencoding-" + embedding_loss,
                    ae_loss.item(),
                    step
                )
                summary_writer.add_scalar(
                    "autoencoder-train/convert-embed-" + embedding_loss,
                    embed_conv_loss.item(),
                    step
                )
                summary_writer.add_scalar(
                    "autoencoder-train/reconstruct-embed-" + embedding_loss,
                    embed_recon_loss.item(),
                    step
                )
                summary_writer.add_scalar(
                    "autoencoder-train/convert-token-" + token_loss,
                    def_conv_loss.item(),
                    step
                )
                summary_writer.add_scalar(
                    "autoencoder-train/reconstruct-token-" + token_loss,
                    def_recon_loss.item(),
                    step
                )
                if recon_flag:
                    summary_writer.add_scalar(
                        "autoencoder-train/recon-convert-embed-" + embedding_loss,
                        embed_conv_recon_loss.item(),
                        step
                    )
                    summary_writer.add_scalar(
                        "autoencoder-train/recon-convert-token-" + token_loss,
                        def_conv_recon_loss.item(),
                        step
                    )
                summary_writer.add_scalar(
                    "autoencoder-train/total-loss",
                    loss.item(),
                    step
                )

            pbar.update(vec.size(0))

        if grad_remains:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        pbar.close()

        ## eval loop
        if dev_file:
            model.eval()
            cur_loss_embed, cur_loss_token = 0.0, 0.0

            with torch.no_grad():
                dev_loss_embed_mse, dev_loss_embed_cos = 0.0, 0.0
                num_tokens, dev_loss_token_acc, dev_loss_token_ce = 0, 0.0, 0.0
                num_samples = 0
                pbar = tqdm.tqdm(
                    desc=f"Eval {epoch}",
                    total=len(dev_dataset),
                    disable=None,
                    leave=False,
                )

                for batch in dev_dataloader:
                    gls = batch["gloss_tensor"].to(device)
                    vec = batch[vec_tensor_key].to(device)

                    _, _, _, _, embed_conv, _ = model(None, gls, mode="inference")
                    _, _, _, _, _, def_conv = model(vec, gls, mode="inference")

                    dev_loss_embed_mse += F.mse_loss(embed_conv,
                                                     vec,
                                                     reduction="none"
                                                     ).mean(1).sum().item()

                    dev_loss_embed_cos += F.cosine_similarity(embed_conv, vec).sum().item()

                    dev_loss_token_ce += F.cross_entropy(def_conv.view(-1, def_conv.size(-1)),
                                                         gls.view(-1),
                                                         reduction="sum",
                                                         ignore_index=model.padding_idx
                                                         ).item()
                    tokens = gls != model.padding_idx
                    num_tokens += tokens.sum().item()
                    dev_loss_token_acc += ((def_conv.argmax(-1) == gls) & tokens).sum().item()
                    num_samples += vec.size(0)

                    pbar.update(vec.size(0))

                # keep track of the average loss on dev set for this epoch
                summary_writer.add_scalar(
                    "autoencoder-dev/convert-embed-mse-per-sample",
                    dev_loss_embed_mse / len(dev_dataset),
                    epoch
                )
                summary_writer.add_scalar(
                    "autoencoder-dev/convert-embed-cos-per-sample",
                    dev_loss_embed_cos / len(dev_dataset),
                    epoch
                )
                summary_writer.add_scalar(
                    "autoencoder-dev/convert-token-ce-per-token",
                    dev_loss_token_ce / num_tokens,
                    epoch
                )
                summary_writer.add_scalar(
                    "autoencoder-dev/convert-token-acc-per-token",
                    dev_loss_token_acc / num_tokens,
                    epoch
                )

                pbar.close()

                if args.embedding_loss == "mse":
                    cur_loss_embed = dev_loss_embed_mse / len(dev_dataset)
                elif args.embedding_loss == "cosine":
                    cur_loss_embed = dev_loss_embed_cos / len(dev_dataset)
                else:
                    raise NotImplementedError

                if args.token_loss == "ce":
                    cur_loss_token = dev_loss_token_ce / num_tokens
                else:
                    raise NotImplementedError

            # 5. save result after evaluation/validation
            if best_loss_embed * 0.999 > cur_loss_embed:
                logger.debug(f"Epoch {epoch}, new best embed loss: {cur_loss_embed:.4f} < "
                             f"{best_loss_embed:.4f}" + f" (x 0.999 = {best_loss_embed * 0.999:.4f})"
                             )
                best_loss_embed = cur_loss_embed
                torch.save(model, save_dir / "model-best-embedding.pt")
                with open(save_dir / "hparams-best-embedding.json", "w") as json_file:
                    hparams = {
                        "learning_rate": learning_rate,
                        "beta1": beta1,
                        "beta2": beta2,
                        "weight_decay": weight_decay,
                    }
                    json.dump(hparams, json_file, indent=2)
                strikes_embed = 0
            else:
                strikes_embed += 1

            if best_loss_token * 0.999 > cur_loss_token:
                logger.debug(f"Epoch {epoch}, new best token loss: {cur_loss_token:.4f} < "
                             f"{best_loss_token:.4f}" + f" (x 0.999 = {best_loss_token * 0.999:.4f})"
                             )
                best_loss_token = cur_loss_token
                torch.save(model, save_dir / "model-best-definition.pt")
                with open(save_dir / "hparams-best-definition.json", "w") as json_file:
                    hparams = {
                        "learning_rate": learning_rate,
                        "beta1": beta1,
                        "beta2": beta2,
                        "weight_decay": weight_decay,
                    }
                    json.dump(hparams, json_file, indent=2)
                strikes_token = 0
            else:
                strikes_token += 1

            if min(strikes_embed, strikes_token) >= patience:
                logger.debug("Stopping early. at epoch " + str(epoch)
                             + ". Token loss did not improve for: "
                             + str(strikes_token)
                             + " epochs, and embed loss did not improve for: "
                             + str(strikes_embed) + " epochs.")
                epochs_range.close()
                break

            model.train()
            # TODO comment out the line below to shuffle training data again?
            # train_dataloader = data.get_dataloader(train_dataset, batch_size=batch_size)

    return best_loss_embed, best_loss_token


def pred(args):
    assert args.test_file is not None, "Missing dataset for test"

    model = OurModel.load(args.save_dir / ("model-best-" + args.pred_direction + ".pt"),
                          map_location=args.device)
    model.eval()

    train_vocab = data.JSONDataset.load(args.save_dir / "train_dataset.pt").vocab
    test_dataset = data.JSONDataset(
        args.test_file, vocab=train_vocab, freeze_vocab=True, maxlen=model.max_len,
        spm_model_name=args.spm_model_path
    )

    pbar = tqdm.tqdm(desc="Predicting " + args.pred_direction, total=len(test_dataset))
    predictions = []

    with torch.no_grad():  # make predictions
        if args.pred_direction == "embedding":
            test_dataloader = data.get_dataloader(test_dataset, batch_size=128, shuffle=False)
            test_data_tensor_key = "gloss_tensor"
            assert test_dataset.has_gloss, "File without glosses, not usable for the task"

            for batch in test_dataloader:
                gls = batch[test_data_tensor_key].to(args.device)
                _, _, _, _, embed_conv, _ = model(None, gls, mode="inference")
                for id, vec in zip(batch["id"], embed_conv.unbind()):
                    predictions.append(
                        {"id": id, args.embedding_arch: vec.view(-1).cpu().tolist()}
                    )
                pbar.update(embed_conv.size(0))

        else:  # equivalent to args.pred_direction == "definition"
            test_dataloader = data.get_dataloader(test_dataset, batch_size=1, shuffle=False)
            test_data_tensor_key = f"{args.embedding_arch}_tensor"
            if args.embedding_arch == "electra":
                assert test_dataset.has_electra, "File without embeddings, not usable for the task"
            else:
                assert test_dataset.has_vecs, "File without embeddings, not usable for the task"
            for batch in test_dataloader:
                vec = batch[test_data_tensor_key].to(args.device)
                batched_sequence = model.pred(word_embedding=vec,
                                              decode_fn=test_dataset.decode,
                                              beam_size=6,
                                              verbose=False)
                for id, gls in zip(batch["id"], test_dataset.decode(batched_sequence)):
                    predictions.append(
                        {"id": id, "gloss": gls}
                    )
                pbar.update(batched_sequence.size(0))

    # 3. dump predictions
    pbar.close()
    with open(args.pred_file, "a") as ostr:
        json.dump(predictions, ostr)


def main(args):
    assert not (args.do_train and args.do_htune), "Conflicting options do_train and do_htune"

    if args.do_train:
        logger.debug("Performing autoencoder training")
        train(
            train_file=args.train_file,
            dev_file=args.dev_file,
            embedding_arch=args.embedding_arch,
            summary_logdir=args.summary_logdir,
            save_dir=args.save_dir,
            device=args.device,
            batch_size=args.batch_size,
            embedding_loss=args.embedding_loss,
            token_loss=args.token_loss,
            label_smoothing=args.label_smoothing,
            skip_flag=args.skip,
            recon_flag=args.reconstruction,
            spm_model_path=args.spm_model_path
        )
    elif args.do_htune:
        logger.debug("Performing autoencoder hyperparameter tuning")
        search_space = get_search_space()

        @skopt.utils.use_named_args(search_space)
        def gp_train(**hparams):
            logger.debug(f"Hyperparams sampled:\n{pprint.pformat(hparams)}")
            best_loss_embed, best_loss_token = train(
                train_file=args.train_file,
                dev_file=args.dev_file,
                embedding_arch=args.embedding_arch,
                summary_logdir=args.summary_logdir,
                save_dir=args.save_dir,
                device=args.device,
                batch_size=args.batch_size,
                embedding_loss=args.embedding_loss,
                token_loss=args.token_loss,
                label_smoothing=hparams["label_smoothing"],
                skip_flag=args.skip,
                recon_flag=args.reconstruction,
                spm_model_path=args.spm_model_path,
                learning_rate=hparams["learning_rate"],
                beta1=min(hparams["beta_a"], hparams["beta_b"]),
                beta2=max(hparams["beta_a"], hparams["beta_b"]),
                weight_decay=hparams["weight_decay"],
                batch_accum=hparams["batch_accum"],
                warmup_len=hparams["warmup_len"],
                n_head=2 ** hparams["n_head_pow"],
                n_layers=hparams["n_layers"],
            )
            return best_loss_embed + best_loss_token  # TODO bring to the same scale

        result = skopt.gp_minimize(gp_train, search_space)
        skopt.dump(result, args.save_dir / "results.pkl", store_objective=False)

    if args.do_pred:
        assert args.pred_direction in {"embedding", "definition"}, \
            "prediction direction must be provided through \"--pred_direction\"."
        logger.debug("Performing autoencoder prediction of " + args.pred_direction)
        pred(args)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
