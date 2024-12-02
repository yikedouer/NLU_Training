# 对用户角色进行embedding
def encode_segment_ids(input_ids, tokenizer):
    sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    usr_id = tokenizer.convert_tokens_to_ids('[USR]')
    sys_id = tokenizer.convert_tokens_to_ids('[SYS]')
    segment_id = []

    usr_flag = 0
    for input_id in input_ids:
        if input_id == usr_id:
            usr_flag = 1
        elif input_id == sys_id:
            usr_flag = 0

        segment_id.append(usr_flag)
        if input_id == sep_id:
            usr_flag = 0
    return segment_id


def truncate_maxlen_with_first_speaker(concate_context, max_context_len):
    # 加入第一个说话角色，如果向前截断，则需要把被截断话术的说话人找出来并放到第一位
    if max_context_len < len(concate_context):
        for i in range(len(concate_context) - max_context_len):
            if concate_context[-(max_context_len + i)] in {'[USR]', '[SYS]'}:
                concate_context[-max_context_len] = concate_context[-(max_context_len + i)]
                break
    return concate_context[-max_context_len:]


def truncate_maxlen_punctuation(concate_context, max_context_len):
    """
    按照标点符号截取语料
    不好用，性能相比直接按照最大长度截断有降低，怀疑是通过标点截断，保留下来的文本更少，导致信息损失
    """
    if max_context_len >= len(concate_context):
        return concate_context
    last_idx = -max_context_len
    for i in range(max_context_len):
        if concate_context[-i - 1] in {',', '，', '？', '！', '、', '!', '.', '。', '：', '(', '（', '~'}:
            last_idx = -i - 1
    if last_idx == -1:
        return concate_context[-max_context_len:]
    else:
        return concate_context[last_idx + 1:]
