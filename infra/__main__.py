"""Pulumi program: Azure NC4as_T4_v3 GPU VM for voice agent."""
import pulumi
from pulumi_azure_native import compute, network, resources

# ── Config ─────────────────────────────────────────────────────────────
config = pulumi.Config()
rg_name = config.get("resourceGroupName") or "rg-voice-agent"
vm_name = config.get("vmName") or "vm-voice-agent"
location = config.get("location") or "eastus"
admin_user = config.get("adminUsername") or "azureuser"
ssh_key = config.require_secret("sshPublicKey")  # set: pulumi config set --secret sshPublicKey "$(cat ~/.ssh/id_rsa.pub)"
repo_url = config.get("repoUrl") or ""           # set: pulumi config set repoUrl "https://github.com/you/repo.git"

# ── Resource Group ─────────────────────────────────────────────────────
rg = resources.ResourceGroup("rg", resource_group_name=rg_name, location=location)

# ── Networking ─────────────────────────────────────────────────────────
vnet = network.VirtualNetwork(
    "vnet",
    resource_group_name=rg.name,
    address_space=network.AddressSpaceArgs(address_prefixes=["10.0.0.0/16"]),
)
nsg = network.NetworkSecurityGroup(
    "nsg",
    resource_group_name=rg.name,
    security_rules=[
        network.SecurityRuleArgs(
            name="ssh", priority=100, direction="Inbound", access="Allow", protocol="Tcp",
            source_port_range="*", destination_port_range="22",
            source_address_prefix="*", destination_address_prefix="*",
        ),
        network.SecurityRuleArgs(
            name="http-https", priority=110, direction="Inbound", access="Allow", protocol="Tcp",
            source_port_range="*", destination_port_ranges=["80", "443"],
            source_address_prefix="*", destination_address_prefix="*",
        ),
        network.SecurityRuleArgs(
            name="voice-ws", priority=120, direction="Inbound", access="Allow", protocol="Tcp",
            source_port_range="*", destination_port_range="8080",
            source_address_prefix="*", destination_address_prefix="*",
        ),
    ],
)
subnet = network.Subnet(
    "subnet",
    resource_group_name=rg.name,
    virtual_network_name=vnet.name,
    address_prefix="10.0.1.0/24",
    network_security_group=network.NetworkSecurityGroupArgs(id=nsg.id),
)
pip = network.PublicIPAddress(
    "pip",
    resource_group_name=rg.name,
    public_ip_allocation_method="Static",
    sku=network.PublicIPAddressSkuArgs(name="Standard"),
)
nic = network.NetworkInterface(
    "nic",
    resource_group_name=rg.name,
    ip_configurations=[
        network.NetworkInterfaceIPConfigurationArgs(
            name="ipcfg",
            private_ip_allocation_method="Dynamic",
            public_ip_address=network.PublicIPAddressArgs(id=pip.id),
            subnet=network.SubnetArgs(id=subnet.id),
        )
    ],
)

# ── VM ──────────────────────────────────────────────────────────────────
vm = compute.VirtualMachine(
    "vm",
    resource_group_name=rg.name,
    location=rg.location,
    hardware_profile=compute.HardwareProfileArgs(vm_size="Standard_NC4as_T4_v3"),
    os_profile=compute.OSProfileArgs(
        computer_name=vm_name,
        admin_username=admin_user,
        linux_configuration=compute.LinuxConfigurationArgs(
            disable_password_authentication=True,
            ssh=compute.SshConfigurationArgs(
                public_keys=[
                    compute.SshPublicKeyArgs(
                        path=f"/home/{admin_user}/.ssh/authorized_keys",
                        key_data=ssh_key,
                    )
                ]
            ),
        ),
    ),
    storage_profile=compute.StorageProfileArgs(
        image_reference=compute.ImageReferenceArgs(
            publisher="Canonical",
            offer="0001-com-ubuntu-server-jammy",
            sku="22_04-lts-gen2",
            version="latest",
        ),
        os_disk=compute.OSDiskArgs(
            name=f"{vm_name}-osdisk",
            caching="ReadWrite",
            create_option="FromImage",
            disk_size_gb=128,
            managed_disk=compute.ManagedDiskParametersArgs(storage_account_type="Premium_LRS"),
        ),
    ),
    network_profile=compute.NetworkProfileArgs(
        network_interfaces=[compute.NetworkInterfaceReferenceArgs(id=nic.id, primary=True)]
    ),
)

# ── NVIDIA GPU Driver Extension ────────────────────────────────────────
gpu_ext = compute.VirtualMachineExtension(
    "nvidia-gpu",
    resource_group_name=rg.name,
    vm_name=vm.name,
    location=rg.location,
    publisher="Microsoft.HpcCompute",
    type_handler_version="1.9",
    type="NvidiaGpuDriverLinux",
)

# ── Bootstrap Script (runs after GPU driver) ───────────────────────────
bootstrap = """
#!/bin/bash
set -e
curl -fsSL https://get.docker.com | bash
usermod -aG docker azureuser
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get install -y nvidia-container-toolkit
systemctl restart docker
mkdir -p /opt/ar-voice-agent
chown azureuser:azureuser /opt/ar-voice-agent
"""

if repo_url:
    bootstrap += f"""
cd /opt/ar-voice-agent
git clone {repo_url} .
chown -R azureuser:azureuser /opt/ar-voice-agent
"""

setup_ext = compute.VirtualMachineExtension(
    "bootstrap",
    resource_group_name=rg.name,
    vm_name=vm.name,
    location=rg.location,
    publisher="Microsoft.Azure.Extensions",
    type_handler_version="2.1",
    type="CustomScript",
    protected_settings={
        "commandToExecute": bootstrap,
    },
    opts=pulumi.ResourceOptions(depends_on=[gpu_ext]),
)

# ── Outputs ─────────────────────────────────────────────────────────────
pulumi.export("public_ip", pip.ip_address)
pulumi.export("ssh_command", pulumi.Output.concat("ssh ", admin_user, "@", pip.ip_address))
pulumi.export("resource_group", rg.name)
pulumi.export("vm_name", vm.name)
pulumi.export("gpu", pulumi.Output.from_input("V100 16GB"))
pulumi.export("gpu_sku", pulumi.Output.from_input("Standard_NC6s_v3"))
