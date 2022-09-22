<?php
$output=null;
$retval=null;

while (true) {
	exec("gcloud alpha compute tpus tpu-vm create tpu-6b --zone europe-west4-a --accelerator-type v3-8 --version tpu-vm-tf-2.6.5", $output, $retval);
	if ($retval!=1)
		break;
	sleep(60);
}
?>