# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from azure.identity import DefaultAzureCredential
from azure.mgmt.managednetworkfabric import ManagedNetworkFabricMgmtClient

"""
# PREREQUISITES
    pip install azure-identity
    pip install azure-mgmt-managednetworkfabric
# USAGE
    python access_control_lists_update_minimum_set_gen.py

    Before run the sample, please set the values of the client ID, tenant ID and client secret
    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,
    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:
    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal
"""


def main():
    client = ManagedNetworkFabricMgmtClient(
        credential=DefaultAzureCredential(),
        subscription_id="subscriptionId",
    )

    response = client.access_control_lists.update(
        resource_group_name="resourceGroupName",
        access_control_list_name="aclOne",
        body={
            "properties": {
                "addressFamily": "ipv4",
                "conditions": [
                    {
                        "action": "allow",
                        "destinationAddress": "1.1.1.2",
                        "destinationPort": "21",
                        "protocol": 6,
                        "sequenceNumber": 4,
                        "sourceAddress": "2.2.2.3",
                        "sourcePort": "65000",
                    }
                ],
            }
        },
    )
    print(response)


# x-ms-original-file: specification/managednetworkfabric/resource-manager/Microsoft.ManagedNetworkFabric/preview/2023-02-01-preview/examples/AccessControlLists_Update_MinimumSet_Gen.json
if __name__ == "__main__":
    main()
