"""
Reference data mirrored from spark_jobs/common.py. Duplicated (not
imported) on purpose: the API and Spark jobs ship as separate Docker images
with no shared filesystem, so this keeps the API container lightweight and
independently deployable at the cost of keeping these in sync by hand if the
taxonomy ever changes - same trade-off api/constants.py documents in the
fraud-detection-spark reference project.
"""

UNIFIED_CATEGORIES = [
    "Benign", "DoS", "Probe/Scan", "Brute Force", "Web Attack", "Botnet",
    "Infiltration", "Privilege Escalation", "Other",
]

DATASETS = ["nsl_kdd", "cicids2017"]

# Verified against the real downloaded KDDTrain+/KDDTest+ files.
NSL_KDD_PROTOCOL_TYPES = ["icmp", "tcp", "udp"]
NSL_KDD_FLAGS = ["OTH", "REJ", "RSTO", "RSTOS0", "RSTR", "S0", "S1", "S2", "S3", "SF", "SH"]
NSL_KDD_SERVICES = [
    "IRC", "X11", "Z39_50", "aol", "auth", "bgp", "courier", "csnet_ns", "ctf", "daytime",
    "discard", "domain", "domain_u", "echo", "eco_i", "ecr_i", "efs", "exec", "finger", "ftp",
    "ftp_data", "gopher", "harvest", "hostnames", "http", "http_2784", "http_443", "http_8001",
    "imap4", "iso_tsap", "klogin", "kshell", "ldap", "link", "login", "mtp", "name",
    "netbios_dgm", "netbios_ns", "netbios_ssn", "netstat", "nnsp", "nntp", "ntp_u", "other",
    "pm_dump", "pop_2", "pop_3", "printer", "private", "red_i", "remote_job", "rje", "shell",
    "smtp", "sql_net", "ssh", "sunrpc", "supdup", "systat", "telnet", "tftp_u", "tim_i", "time",
    "urh_i", "urp_i", "uucp", "uucp_path", "vmnet", "whois",
]

# The full 41-feature vector the NSL-KDD PipelineModel expects, in order.
NSL_KDD_NUMERIC_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
NSL_KDD_CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

# A hand-picked subset of the 41 features exposed on the live-demo form -
# the rest default to a "typical, unremarkable connection" baseline in
# inference.py rather than forcing a user to fill in all 41 fields. Chosen
# from the trained model's real feature_importances (service_idx and
# src_bytes alone account for ~78% of NSL-KDD's importance - see the README).
NSL_KDD_FORM_FIELDS = [
    "protocol_type", "service", "flag", "duration", "src_bytes", "dst_bytes",
    "count", "srv_count", "num_failed_logins", "logged_in", "serror_rate",
    "rerror_rate", "land", "wrong_fragment",
]
NSL_KDD_DEFAULTS = {f: 0.0 for f in NSL_KDD_NUMERIC_FEATURES}
NSL_KDD_DEFAULTS.update({
    "logged_in": 1.0, "same_srv_rate": 1.0, "dst_host_count": 1.0,
    "dst_host_srv_count": 1.0, "dst_host_same_srv_rate": 1.0,
})
NSL_KDD_CATEGORICAL_DEFAULTS = {"protocol_type": "tcp", "service": "http", "flag": "SF"}

# CICIDS2017: the full feature vector the PipelineModel expects, in order
# (mirrors common.CICIDS2017_NUMERIC_FEATURES; no categorical features -
# this CSV release has no protocol column, see spark_jobs/common.py).
CICIDS2017_NUMERIC_FEATURES = [
    "destination_port", "flow_duration", "total_fwd_packets", "total_backward_packets",
    "total_length_of_fwd_packets", "total_length_of_bwd_packets",
    "fwd_packet_length_max", "fwd_packet_length_min", "fwd_packet_length_mean", "fwd_packet_length_std",
    "bwd_packet_length_max", "bwd_packet_length_min", "bwd_packet_length_mean", "bwd_packet_length_std",
    "flow_bytes_per_s", "flow_packets_per_s",
    "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
    "fwd_iat_total", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max", "fwd_iat_min",
    "bwd_iat_total", "bwd_iat_mean", "bwd_iat_std", "bwd_iat_max", "bwd_iat_min",
    "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
    "fwd_header_length", "bwd_header_length",
    "fwd_packets_per_s", "bwd_packets_per_s",
    "min_packet_length", "max_packet_length", "packet_length_mean", "packet_length_std", "packet_length_variance",
    "fin_flag_count", "syn_flag_count", "rst_flag_count", "psh_flag_count", "ack_flag_count",
    "urg_flag_count", "cwe_flag_count", "ece_flag_count",
    "down_up_ratio", "average_packet_size", "avg_fwd_segment_size", "avg_bwd_segment_size",
    "fwd_avg_bytes_bulk", "fwd_avg_packets_bulk", "fwd_avg_bulk_rate",
    "bwd_avg_bytes_bulk", "bwd_avg_packets_bulk", "bwd_avg_bulk_rate",
    "subflow_fwd_packets", "subflow_fwd_bytes", "subflow_bwd_packets", "subflow_bwd_bytes",
    "init_win_bytes_forward", "init_win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "active_mean", "active_std", "active_max", "active_min",
    "idle_mean", "idle_std", "idle_max", "idle_min",
]

# Chosen from the trained model's real feature_importances (bwd_packet_length_std,
# destination_port, and bwd_packet_length_mean alone account for ~80% - see README).
CICIDS2017_FORM_FIELDS = [
    "destination_port", "flow_duration", "total_fwd_packets", "total_backward_packets",
    "total_length_of_fwd_packets", "total_length_of_bwd_packets",
    "fwd_packet_length_mean", "fwd_packet_length_std", "bwd_packet_length_mean", "bwd_packet_length_std",
    "flow_bytes_per_s", "flow_packets_per_s", "syn_flag_count", "ack_flag_count",
]
CICIDS2017_DEFAULTS = {f: 0.0 for f in CICIDS2017_NUMERIC_FEATURES}
