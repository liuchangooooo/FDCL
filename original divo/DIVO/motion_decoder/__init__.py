from DIVO.motion_decoder.tcnmd import TCNMotionDecoder

def get_motion_decoder(**motion_decoder_cfg):
    target = motion_decoder_cfg["_target_"]
    if target == "tcn":
        motion_decoder = TCNMotionDecoder(
            **motion_decoder_cfg
        )
    return motion_decoder 